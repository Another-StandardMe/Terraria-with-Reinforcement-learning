import torch
import torch.multiprocessing as mp
import numpy as np
import os

from queue import Empty
from TerrariaPPO import PPO
from TerrariaENV import TerrariaEnv
import time


class Args:
    """ 定义超参数 """
    model_path = "10000best.pt"
    seq_len = 8

    # Swin 模型参数
    # embed_dim = 64 # Swin Transformer 3D输出特征维度
    # hidden_dim = 128
    # dropout_rate = 0.1
    # num_frames = 8
    # img_size = 224

    #CNN-Transformer模型参数
    img_feature_dim = 128
    transformer_dim = 64
    hidden_width = 128
    transformer_heads = 2
    transformer_layers = 4
    dropout_rate = 0.1

    # **PPO 训练参数**d
    gamma = 0.994
    lamda = 0.95
    epsilon = 0.5
    K_epochs = 10
    entropy_coef = 0.0001
    max_episodes = 3e4
    batch_size = 4

    # **优化器 & 训练**
    lr_a = 2e-5
    lr_c = 6e-5
    max_epochs = 1000
    save_step = 15000
    clip_value = 0.4

    # **Worker-Learner 并行**
    num_workers = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_args():
    return Args()


class Worker(mp.Process):
    """ 采集数据并发送给 Learner """

    def __init__(self, worker_id, data_queue, param_queue, epoch_sync_event, exit_event, args):
        super(Worker, self).__init__()
        self.worker_id = worker_id
        self.data_queue = data_queue
        self.param_queue = param_queue
        self.epoch_sync_event = epoch_sync_event
        self.exit_event = exit_event
        self.args = args

    def run(self):
        print(f"🚀 Worker {self.worker_id} 初始化环境")
        env = TerrariaEnv(self.args.model_path, self.args.seq_len)
        model = PPO(self.args)
        model.actor.to(self.args.device)  # ✅ 确保模型在 CUDA

        for epoch in range(self.args.max_epochs):
            if self.exit_event.is_set():
                break

            print(f"🎯 Worker {self.worker_id} 开始 Epoch {epoch + 1}/{self.args.max_epochs}")

            obs = env.reset()
            while obs is None:
                time.sleep(1)
                obs = env._get_observation()

            done = False

            while not done:
                if self.exit_event.is_set():
                    break

                # 获取最新策略
                try:
                    params = self.param_queue.get_nowait()  # 获取最新的模型参数
                    model.actor.load_state_dict(params)
                    model.actor.to(self.args.device)
                except Empty:
                    pass

                action, log_prob = model.get_action(obs.to(self.args.device))  # ✅ 确保 obs 在 GPU

                next_obs, reward, done, _ = env.step(action)

                if next_obs is None:
                    continue

                # 存储数据（单样本）
                try:
                    self.data_queue.put_nowait((
                        obs.cpu().clone(),  # 确保复制到 CPU
                        torch.tensor(action, dtype=torch.long).cpu().clone(),
                        torch.tensor(log_prob, dtype=torch.float).cpu().clone(),
                        torch.tensor(reward, dtype=torch.float).cpu().clone(),
                        next_obs.cpu().clone(),  # 确保复制到 CPU
                        torch.tensor(float(done), dtype=torch.float).cpu().clone()
                    ))
                    self.epoch_sync_event.set()  # 通知 Learner 训练
                except:
                    print(f"⚠️ Worker {self.worker_id}: 数据队列满，跳过存储")

                obs = next_obs

            print(f"✅ Worker {self.worker_id} 结束 Epoch {epoch + 1}")
            time.sleep(1)

        self.data_queue.put(("TERMINATE", self.worker_id))  # ✅ 结构化终止信号
        print(f"🚀 Worker {self.worker_id} 训练完成")


class Learner(mp.Process):
    """ Learner 端：持续训练 """
    def __init__(self, model, data_queue, param_queue, epoch_sync_event, exit_event, args):
        super(Learner, self).__init__()
        self.model = model
        self.data_queue = data_queue
        self.param_queue = param_queue
        self.epoch_sync_event = epoch_sync_event
        self.exit_event = exit_event
        self.args = args
        self.total_samples = 0  # 记录训练的数据量
        self.active_workers = self.args.num_workers  # 记录当前活跃的 workers 数量

    def run(self):
        self.model.actor.to(self.args.device)  # 确保模型在 GPU
        self.model.critic.to(self.args.device)  # 确保模型在 GPU

        print("📢 Learner 开始训练")

        batch_samples = []  # 用于收集 batch
        while self.active_workers > 0:
            try:
                data = self.data_queue.get(timeout=5)
                if isinstance(data, tuple) and data[0] == "TERMINATE":
                    self.active_workers -= 1
                    print(f"⚠️ Learner 收到终止信号，Worker {data[1]} 已退出")
                    continue

                batch_samples.append(data)
                self.total_samples += 1

                if len(batch_samples) >= 4:
                    self.model.update(batch_samples)
                    batch_samples = []

                    if self.total_samples % self.args.save_step == 0:
                        save_path = f"checkpoints/Terraria_checkpoint_{self.total_samples}.pth"
                        os.makedirs("checkpoints", exist_ok=True)
                        torch.save(self.model.actor.state_dict(), save_path)
                        print(f"💾 定期保存模型: {save_path} (训练数据: {self.total_samples})")

                        # 发送最新模型参数之前先转换到 CPU
                        self.param_queue.put({k: v.cpu() for k, v in self.model.actor.state_dict().items()})

            except Empty:
                print("⚠️ 数据队列为空，等待数据...")
                time.sleep(1)

        save_path = "checkpoints/Terraria_final_model.pth"
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(self.model.actor.state_dict(), save_path)
        print(f"💾 最终模型已保存: {save_path}")

        print("✅ Learner 训练完成，通知 Worker 退出")
        self.exit_event.set()



def train():
    mp.set_start_method("spawn", force=True)
    args = get_args()

    agent = PPO(args)
    data_queue = mp.Queue(maxsize=6000)
    param_queue = mp.Queue()
    epoch_sync_event = mp.Event()
    exit_event = mp.Event()

    workers = [Worker(i, data_queue, param_queue, epoch_sync_event, exit_event, args) for i in range(args.num_workers)]
    learner = Learner(agent, data_queue, param_queue, epoch_sync_event, exit_event, args)

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




