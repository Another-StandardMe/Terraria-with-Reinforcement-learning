import torch
import torch.multiprocessing as mp
import numpy as np
import os
from ppo_categorical import PPO  # ✅ 替换 PPO_discrete
from queue import Empty
from Game_Env import TerrariaEnv
import time

class Args:
    """ 定义超参数 """
    model_path = "E:/terraria_project/after_training_weight/10000best.pt"
    seq_len = 8

    # **模型参数**
    img_feature_dim = 128
    transformer_dim = 64
    hidden_width = 128
    transformer_heads = 2
    transformer_layers = 3
    dropout_rate = 0.1

    # **PPO 训练参数**
    batch_size = 16
    mini_batch_size = 8
    gamma = 0.99
    lamda = 0.95
    epsilon = 0.2
    K_epochs = 10
    entropy_coef = 0.01

    # **优化器 & 训练**
    lr_a = 3e-4
    lr_c = 1e-5
    max_epochs = 3
    save_interval = 1
    use_grad_clip = True
    use_lr_decay = True
    use_adv_norm = True

    # **Worker-Learner 并行**
    num_workers = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    return Args()

class Worker(mp.Process):
    """ Worker 进程：与环境交互，采样数据 """
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
        model = PPO(self.args)  # ✅ 替换 PPO_discrete
        model.actor.to(self.args.device)

        for epoch in range(self.args.max_epochs):
            if self.exit_event.is_set():
                break

            obs = env.reset()
            while obs is None:
                time.sleep(1)
                obs = env._get_observation()

            model.actor.reset_cache()
            done = False

            while not done:
                if self.exit_event.is_set():
                    break

                try:
                    params = self.param_queue.get_nowait()
                    model.actor.load_state_dict(params)
                except Empty:
                    pass

                action, log_prob = model.get_action(obs)
                next_obs, reward, done, _ = env.step(action)

                if next_obs is None:
                    continue

                if not isinstance(obs, torch.Tensor):
                    obs = torch.tensor(np.array(obs), dtype=torch.float32, device=self.args.device)

                if not isinstance(next_obs, torch.Tensor):
                    next_obs = torch.tensor(np.array(next_obs), dtype=torch.float32, device=self.args.device)

                self.data_queue.put((obs, action, log_prob, reward, next_obs, done))
                self.data_queue.put((
                    obs, action, log_prob, reward, next_obs, done
                ))

                obs = next_obs

            self.epoch_sync_event.set()
            time.sleep(2)
        print(f"🚀 Worker {self.worker_id} 结束训练")

class Learner(mp.Process):
    """ Learner 进程：训练 PPO 并更新 Worker 参数 """
    def __init__(self, model, data_queue, param_queue, epoch_sync_event, exit_event, args):
        super(Learner, self).__init__()
        self.model = model
        self.data_queue = data_queue
        self.param_queue = param_queue
        self.epoch_sync_event = epoch_sync_event
        self.exit_event = exit_event
        self.args = args
        self.replay_buffer = []

    def run(self):
        self.model.actor.to(self.args.device)
        self.model.critic.to(self.args.device)

        for epoch in range(self.args.max_epochs):
            if self.exit_event.is_set():
                break

            self.epoch_sync_event.wait()
            self.epoch_sync_event.clear()

            for _ in range(self.args.batch_size):
                try:
                    data = self.data_queue.get(timeout=5)
                    self.replay_buffer.append(data)
                except Empty:
                    break

            if self.replay_buffer:
                self._train()

            self.param_queue.put(self.model.actor.cpu().state_dict())
            self.model.actor.to(self.args.device)

            if epoch % self.args.save_interval == 0:
                save_path = f"checkpoints/Terraria_model_epoch_{epoch}.pth"
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(self.model.actor.state_dict(), save_path)

        print("✅ Learner 训练完成，通知 Worker 退出")
        self.exit_event.set()

    def _train(self):
        if not self.replay_buffer:
            return

        obs, actions, log_probs, rewards, next_obs, dones = zip(*self.replay_buffer)
        self.replay_buffer = []
        self.model.update((obs, actions, log_probs, rewards, next_obs, dones))

def train():
    mp.set_start_method("spawn", force=True)
    args = get_args()

    agent = PPO(args)  # ✅ 替换 PPO_discrete
    data_queue = mp.Queue()
    param_queue = mp.Queue()
    epoch_sync_event = mp.Event()
    exit_event = mp.Event()

    workers = [Worker(i, data_queue, param_queue, epoch_sync_event, exit_event, args) for i in range(args.num_workers)]
    learner = Learner(agent, data_queue, param_queue, epoch_sync_event, exit_event, args)

    for w in workers:
        w.start()
    learner.start()
    learner.join()

    for w in workers:
        w.join()

    print("✅ 训练完成，所有进程已终止")

if __name__ == "__main__":
    train()
