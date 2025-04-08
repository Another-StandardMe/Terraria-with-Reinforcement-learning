import torch
import torch.multiprocessing as mp
import os
import time
from queue import Empty

from TerrariaPPO import PPO
from TerrariaENV import TerrariaEnv

class Args:
    """ 定义超参数 """
    model_path = "10000best.pt"
    seq_len = 4

    # CNN-Transformer模型参数
    img_feature_dim = 128
    transformer_dim = 128
    hidden_width = 64
    transformer_heads = 4
    transformer_layers = 1
    dropout_rate = 0.1

    # PPO 训练参数
    gamma = 0.90
    lamda = 0.95  # 虽然 PPO 更新中不使用，但保留以便以后扩展
    epsilon = 0.5
    K_epochs = 10
    entropy_coef = 0.01
    max_train_steps = 3e5
    batch_size = 4

    # 优化器 & 训练
    lr_a = 2e-4
    lr_c = 6e-4
    max_epochs = 10000
    save_step = 1000000
    clip_value = 0.4

    # Worker-Learner 并行
    num_workers = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_args():
    return Args()

class Worker(mp.Process):
    """ 采集数据并发送给 Learner """
    def __init__(self, worker_id, data_queue, param_queue, epoch_sync_event, epoch_end_event, exit_event, args):
        super(Worker, self).__init__()
        self.worker_id = worker_id
        self.data_queue = data_queue
        self.param_queue = param_queue
        self.epoch_sync_event = epoch_sync_event  # 用于通知 Learner 新数据到位
        self.epoch_end_event = epoch_end_event      # 用于等待 Learner 完成本轮训练
        self.exit_event = exit_event
        self.args = args

    def run(self):
        print(f"🚀 Worker {self.worker_id} 初始化环境")
        env = TerrariaEnv(self.args.seq_len)
        # 实例化 PPO（内部自动执行正交初始化）
        model = PPO(self.args)
        model.actor.to(self.args.device)

        for epoch in range(self.args.max_epochs):
            if self.exit_event.is_set():
                break

            print(f"🎯 Worker {self.worker_id} 开始 Epoch {epoch + 1}/{self.args.max_epochs}")

            obs = env.reset()
            while obs is None:
                time.sleep(1)
                obs = env._get_observation()

            boss = env.check()
            if boss:  # 如果 boss == True
                print("⏸️ 环境出错，重置环境...")
                obs = env.reset()

            # 如果 obs 是单样本（形状 [seq_len, 3, H, W]），增加 batch 维度
            if obs.ndim == 4:
                obs = obs.unsqueeze(0)

            done = False

            while not done:
                if self.exit_event.is_set():
                    break

                # 尝试获取最新的模型参数
                try:
                    params = self.param_queue.get_nowait()
                    model.actor.load_state_dict(params)
                    model.actor.to(self.args.device)
                except Empty:
                    pass

                # 获取动作和 log_prob（get_action 返回的是列表，此处转换为张量并补齐 batch 维度）
                action, log_prob = model.get_action(obs.to(self.args.device))
                action_tensor = torch.tensor(action, dtype=torch.long)
                if action_tensor.ndim == 1:
                    action_tensor = action_tensor.unsqueeze(0)
                log_prob_tensor = torch.tensor(log_prob, dtype=torch.float)
                if log_prob_tensor.ndim == 0 or log_prob_tensor.ndim == 1:
                    log_prob_tensor = log_prob_tensor.unsqueeze(0)

                next_obs, reward, done, _ = env.step(action)
                if next_obs is None:
                    continue
                if next_obs.ndim == 4:
                    next_obs = next_obs.unsqueeze(0)

                try:
                    # 构造单个样本数据（每个元素都带有 batch 维度）
                    sample = (
                        obs.cpu().clone(),                      # [1, seq_len, 3, H, W]
                        action_tensor.cpu().clone(),            # [1, 2]
                        log_prob_tensor.cpu().clone(),          # [1] 或 [1,1]
                        torch.tensor(reward, dtype=torch.float).unsqueeze(0).cpu().clone(),  # [1]
                        next_obs.cpu().clone(),                 # [1, seq_len, 3, H, W]
                        torch.tensor(float(done), dtype=torch.float).unsqueeze(0).cpu().clone()  # [1]
                    )
                    self.data_queue.put_nowait(sample)
                    self.epoch_sync_event.set()  # 通知 Learner 数据已到位
                except Exception as e:
                    print(f"⚠️ Worker {self.worker_id}: 数据队列满，跳过存储，错误信息: {e}")

                obs = next_obs

            print(f"✅ Worker {self.worker_id} 结束 Epoch {epoch + 1}")
            env.close()
            # 发送本轮结束信号
            self.data_queue.put(("EPOCH_END", self.worker_id))
            print(f"⚠️ Worker {self.worker_id} 等待 Learner 完成当前 Epoch 的训练...")
            self.epoch_end_event.wait()  # 等待 Learner 通知本轮 epoch 数据处理完毕
            self.epoch_end_event.clear()
            print("----- 接下来重置环境 -----")

        self.data_queue.put(("TERMINATE", self.worker_id))
        print(f"🚀 Worker {self.worker_id} 训练完成")

class Learner(mp.Process):
    """ Learner 端：持续训练 """
    def __init__(self, model, data_queue, param_queue, epoch_sync_event, epoch_end_event, exit_event, args):
        super(Learner, self).__init__()
        self.model = model
        self.data_queue = data_queue
        self.param_queue = param_queue
        self.epoch_sync_event = epoch_sync_event  # 用于通知新数据到达
        self.epoch_end_event = epoch_end_event      # 用于通知 Worker 当前 epoch 完成训练
        self.exit_event = exit_event
        self.args = args
        self.total_samples = 0
        self.active_workers = self.args.num_workers

    def run(self):
        self.model.actor.to(self.args.device)
        self.model.critic.to(self.args.device)
        print("📢 Learner 开始训练")

        while self.active_workers > 0 or not self.data_queue.empty():
            try:
                data = self.data_queue.get(timeout=5)
                if isinstance(data, tuple):
                    # 处理 Worker 发来的信号
                    if data[0] == "TERMINATE":
                        self.active_workers -= 1
                        print(f"⚠️ Learner 收到终止信号，Worker {data[1]} 已退出")
                        break
                        # continue # 等待多个worker结束时使用

                    elif data[0] == "EPOCH_END":
                        print(f"⚠️ Learner 收到 Epoch 结束信号，Worker {data[1]} 本轮数据训练完成")
                        # 通知对应 Worker 当前 epoch 训练完成
                        self.epoch_end_event.set()
                        continue

                # 直接传入单个样本数据（保证数据各元素带有 batch 维度）
                self.model.update(data)
                self.total_samples += 1

                if self.total_samples % self.args.save_step == 0:
                    save_path = f"checkpoints/Terraria_checkpoint_{self.total_samples}.pth"
                    os.makedirs("checkpoints", exist_ok=True)
                    torch.save(self.model.actor.state_dict(), save_path)
                    print(f"💾 定期保存模型: {save_path} (训练数据: {self.total_samples})")
                    self.param_queue.put({k: v.cpu() for k, v in self.model.actor.state_dict().items()})
            except Empty:
                print("⚠️ 数据队列为空，等待数据...")
                time.sleep(1)

        save_path = "checkpoints/Terraria_final_model.pth"
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            "actor": self.model.actor.state_dict(),
            "critic": self.model.critic.state_dict()
        }, save_path)
        print(f"💾 最终模型已保存: {save_path}")
        print("✅ Learner 训练完成，通知 Worker 退出")
        self.exit_event.set()

def train():
    mp.set_start_method("spawn", force=True)
    args = get_args()

    # 新增 epoch_end_event，用于同步每个 epoch 的结束
    epoch_sync_event = mp.Event()
    epoch_end_event = mp.Event()
    exit_event = mp.Event()

    # 实例化 PPO（内部自动执行正交初始化）
    agent = PPO(args)
    data_queue = mp.Queue(maxsize=6000)
    param_queue = mp.Queue()

    workers = [Worker(i, data_queue, param_queue, epoch_sync_event, epoch_end_event, exit_event, args)
               for i in range(args.num_workers)]
    learner = Learner(agent, data_queue, param_queue, epoch_sync_event, epoch_end_event, exit_event, args)

    for w in workers:
        w.start()
    learner.start()

    learner.join()
    exit_event.set()
    for w in workers:
        w.join()

    print("✅ 训练完成")

if __name__ == "__main__":
    train()
