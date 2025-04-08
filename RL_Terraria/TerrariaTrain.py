import torch
import torch.multiprocessing as mp
import os
import time
from queue import Empty
import numpy as np

from TerrariaPPO import PPO
from TerrariaENV import TerrariaEnv

class Args:
    """ 定义超参数 """
    # YOLO 模型加载
    # model_path = "10000best.pt"
    date = 250403
    seq_len = 4

    # CNN-Transformer模型参数
    img_feature_dim = 128
    transformer_dim = 128
    hidden_width = 128
    transformer_heads = 2
    transformer_layers = 2
    dropout_rate = 0.1

    # PPO 训练参数
    gamma = 0.98
    lamda = 0.95
    epsilon = 0.4
    K_epochs = 3
    entropy_coef = 0.005
    max_train_steps = 450 # 500epoch:4e5, 1800epoch:15e5
    batch_size = 4

    # 优化器 & 训练
    lr_a = 1e-3
    lr_c = 2e-3
    max_episodes = 150 # 300episode = 34329.365225076675s = 9.53 hour   28175.96843
    save_step = 2e4
    clip_value = 0.4

    # 是否使用 Adam eps 参数
    set_adam_eps = True

    # 是否启用梯度裁剪（用于训练稳定性）
    use_grad_clip = True

    # 是否启用学习率衰减（CosineAnnealing）
    use_lr_decay = True

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

        for epoch in range(self.args.max_episodes):
            if self.exit_event.is_set():
                break

            trajectory = {
                "states": [], "actions": [], "log_probs": [], "rewards": [], "next_states": [], "dones": []
            }

            print(f"🎯 Worker {self.worker_id} 开始 Epoch {epoch + 1}/{self.args.max_episodes}")

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
            ep_length = 0
            ep_reward = 0.0

            while not done:
                if self.exit_event.is_set():
                    break
                # start = time.time()

                # 尝试获取最新的模型参数
                try:
                    params = self.param_queue.get_nowait()
                    model.actor.load_state_dict(params)
                    model.actor.to(self.args.device)

                    print(f"💠 第[{ep_length}]次 接收新参数......")

                    # for name, param in model.actor.state_dict().items():
                    #     if not torch.equal(param.cpu(), params[name].cpu()):
                    #         print(f"⚠️ Worker {self.worker_id}: 参数 `{name}` 加载失败")
                    #     else:
                    #         print(f"✅ Worker {self.worker_id}: 参数 `{name}` 匹配")
                except Empty:
                    pass

                # 如果 obs 为 None，重新获取状态（跳过动作）
                if obs is None:
                    obs = env._get_observation()
                    if obs is None:
                        time.sleep(0.001)
                        continue
                    if obs.ndim == 4:
                        obs = obs.unsqueeze(0)

                # 获取动作和 log_prob（get_action 返回的是列表，此处转换为张量并补齐 batch 维度）
                action, log_prob = model.get_action(obs.to(self.args.device))

                action_tensor = torch.tensor(action, dtype=torch.long)
                if action_tensor.ndim == 1:
                    action_tensor = action_tensor.unsqueeze(0)
                log_prob_tensor = torch.tensor(log_prob, dtype=torch.float)
                if log_prob_tensor.ndim == 0 or log_prob_tensor.ndim == 1:
                    log_prob_tensor = log_prob_tensor.unsqueeze(0)

                next_obs, reward, done, _ = env.step(action)

                # 如果无效，忽略该动作，重新获取 obs
                if next_obs is None:
                    print(f"⚠️ Worker {self.worker_id}: 当前动作忽略 (截图失败)")
                    obs = None
                    continue

                if next_obs.ndim == 4:
                    next_obs = next_obs.unsqueeze(0)

                try:
                    # 构造单个样本数据（每个元素都带有 batch 维度）
                    # sample = (
                    #     obs.cpu().clone(),                      # [1, seq_len, 3, H, W]
                    #     action_tensor.cpu().clone(),            # [1, 2]
                    #     log_prob_tensor.cpu().clone(),          # [1] 或 [1,1]
                    #     torch.tensor(reward, dtype=torch.float).unsqueeze(0).cpu().clone(),  # [1]
                    #     next_obs.cpu().clone(),                 # [1, seq_len, 3, H, W]
                    #     torch.tensor(float(done), dtype=torch.float).unsqueeze(0).cpu().clone()  # [1]
                    # )
                    # self.data_queue.put_nowait(sample)
                    # self.epoch_sync_event.set()  # 通知 Learner 数据已到位

                    trajectory["states"].append(obs.cpu().clone())
                    trajectory["actions"].append(action_tensor.cpu().clone())
                    trajectory["log_probs"].append(log_prob_tensor.cpu().clone())
                    trajectory["rewards"].append(torch.tensor(reward, dtype=torch.float).unsqueeze(0).cpu().clone())
                    trajectory["next_states"].append(next_obs.cpu().clone())
                    trajectory["dones"].append(torch.tensor(float(done), dtype=torch.float).unsqueeze(0).cpu().clone())

                except Exception as e:
                    print(f"⚠️ Worker {self.worker_id}: 数据队列满，跳过存储，错误信息: {e}")

                ep_length += 1
                ep_reward += reward
                # print(f"reward: {reward}, ep_reward: {ep_reward}")
                # print(f"worker{ep_length}:{time.time()-start}")
                obs = next_obs

            print(f"✅ Worker {self.worker_id} 结束 Epoch {epoch + 1} :{ep_length}ep | ep_reward: {ep_reward}")
            env.close()
            # 发送本轮结束信号
            self.data_queue.put(("TRAJECTORY", trajectory))
            self.data_queue.put(("EPOCH_END", self.worker_id, epoch + 1, ep_length, ep_reward, _))
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

        with open(f"epoch_record_{self.args.date}.txt", "a") as f:
                        f.write("=" * 60 + "\n")

        while self.active_workers > 0 or not self.data_queue.empty():
            try:
                data = self.data_queue.get(timeout=5)
                if isinstance(data, tuple):

                    if data[0] == "TRAJECTORY":
                        _, traj = data
                        self.model.update_trajectory(traj)
                        self.total_samples += 1

                        # 清空 param_queue 中的旧参数，确保队列只保存最新参数
                        while not self.param_queue.empty():
                            try:
                                self.param_queue.get_nowait()
                            except Empty:
                                break
                        # 更新最新参数到 param_queue
                        self.param_queue.put({k: v.cpu() for k, v in self.model.actor.state_dict().items()})
                        print(f"💠 Learner 发送参数......")

                        if self.total_samples % 30 == 0:
                            save_path = f"checkpoints/Terraria_checkpoint_{self.total_samples}.pth"
                            os.makedirs("checkpoints", exist_ok=True)
                            torch.save(self.model.actor.state_dict(), save_path)
                            torch.cuda.empty_cache()
                            print(f"💾 定期保存模型: {save_path} (训练数据: {self.total_samples})")

                        continue

                    # 处理 Worker 发来的信号
                    if data[0] == "TERMINATE":
                        self.active_workers -= 1
                        print(f"⚠️ Learner 收到终止信号，Worker {data[1]} 已退出")
                        break
                        # continue # 等待多个worker结束时使用

                    elif data[0] == "EPOCH_END":
                        print(f"⚠️ Learner 收到 Epoch 结束信号，Worker {data[1]} 本轮数据训练完成")
                        # 解析 Worker 发来的 epoch 结束信号
                        # data 格式为: ("EPOCH_END", worker_id, epoch_num, ep_length, ep_reward)
                        _, worker_id, epoch_num, ep_length, ep_reward, kill = data

                        # 如果 PPO 在 update() 中记录了损失和 Q 值指标，
                        # 可以在这里取最近一段的平均值，作为本 epoch 的指标
                        if len(self.model.ep_avg_losses) > 0:
                            ep_avg_losses = np.mean(self.model.ep_avg_losses)
                        else:
                            ep_avg_losses = 0.0
                        if len(self.model.ep_avg_qs) > 0:
                            ep_avg_qs = np.mean(self.model.ep_avg_qs)
                        else:
                            ep_avg_qs = 0.0

                        # 将指标写入 txt 文件，格式如：
                        # epoch 1: ep_length, ep_avg_losses, ep_avg_qs, ep_reward, kill:{kill}
                        with open(f"epoch_record_{self.args.date}.txt", "a") as f:
                            f.write(
                                f"epoch {epoch_num}: step:{ep_length}, avg_policy_loss:{ep_avg_losses:.8f}, avg_q_value:{ep_avg_qs:.8f}, reward:{ep_reward:.4f}, kill: {kill}\n")
                        print(f"📄 记录 epoch {epoch_num} 指标到文件")

                        # 可选：清空 PPO 中本 epoch的指标（如果需要按 epoch 分开统计）
                        self.model.ep_avg_losses.clear()
                        self.model.ep_avg_qs.clear()
                        self.model.resetMemory()

                        # 通知对应 Worker 当前 epoch 训练完成
                        self.epoch_end_event.set()
                        continue

                # 直接传入单个样本数据（保证数据各元素带有 batch 维度）
                # self.model.update(data)
                # self.total_samples += 1
                #
                # if self.active_workers > 0 and self.total_samples % 100 == 0:
                #     # 清空 param_queue 中的旧参数，确保队列只保存最新参数
                #     while not self.param_queue.empty():
                #         try:
                #             self.param_queue.get_nowait()
                #         except Empty:
                #             break
                #     # 更新最新参数到 param_queue
                #     self.param_queue.put({k: v.cpu() for k, v in self.model.actor.state_dict().items()})
                #     print(f"💠 第[{self.total_samples}]次 发送参数......")

                # if self.total_samples % self.args.save_step == 0:
                #     save_path = f"checkpoints/Terraria_checkpoint_{self.total_samples}.pth"
                #     os.makedirs("checkpoints", exist_ok=True)
                #     torch.save(self.model.actor.state_dict(), save_path)
                #     print(f"💾 定期保存模型: {save_path} (训练数据: {self.total_samples})")


            except Empty:
                # print("⚠️ 数据队列为空，等待数据...")
                time.sleep(1)

        save_path = "checkpoints/Terraria_final_model_005.pth"
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            "actor": self.model.actor.state_dict(),
            "critic": self.model.critic.state_dict()
        }, save_path)
        print(f"💾 最终模型已保存: {save_path}")
        print("✅ Learner 训练完成，通知 Worker 退出")
        self.exit_event.set()

def train():
    start = time.time()

    mp.set_start_method("spawn", force=True)
    args = get_args()

    # 新增 epoch_end_event，用于同步每个 epoch 的结束
    epoch_sync_event = mp.Event()
    epoch_end_event = mp.Event()
    exit_event = mp.Event()

    # 实例化 PPO（内部自动执行正交初始化）
    agent = PPO(args)
    data_queue = mp.Queue(maxsize=6000)
    param_queue = mp.Queue(maxsize=1)

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

    print(f"✅ 训练完成 time: {time.time()-start}s...")

if __name__ == "__main__":
    train()
