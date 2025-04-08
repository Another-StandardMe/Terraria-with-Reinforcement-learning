import torch
import time
from G_PolicyNet import GTrXLPolicyNet
from TerrariaENV import TerrariaEnv
from TerrariaTrain import get_args


def ppo_model_test(episodes):
    args = get_args()
    args.model_path = "D:/RL_Terraria/Project_TAI/newest/checkpoints/004-005/Terraria_checkpoint_150.pth"
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = TerrariaEnv(seq_len=args.seq_len, verbose=False)
    policy_net = GTrXLPolicyNet().to(args.device)
    policy_net.load_state_dict(torch.load(args.model_path, map_location=args.device))
    policy_net.eval()
    print("✅ 成功加载策略模型参数")

    for ep_num in range(episodes):
        obs = env.reset()
        while obs is None:
            print("⚠️ 初始状态采集失败，重新采集中...")
            time.sleep(0.5)
            obs = env._get_observation()

        # [SEQ_LEN, 1, H, W] -> [1, SEQ_LEN, 1, H, W]
        obs = obs.unsqueeze(0)
        policy_net.reset_memory()

        total_reward = 0
        step_count = 0
        done = False

        print(f"开始第 {ep_num + 1} 轮测试...")

        while not done:
            # 只要状态为 None，重新获取，丢弃前一帧动作逻辑
            if obs is None:
                obs = env._get_observation()
                if obs is None:
                    print("⚠️ 状态获取失败，跳过该帧")
                    time.sleep(0.05)
                    continue
                obs = obs.unsqueeze(0)

            with torch.no_grad():
                actions, _ = policy_net.sample_action(obs.to(args.device))

            action = actions.cpu().numpy().squeeze(0).tolist()

            # 执行动作
            next_obs, reward, done, kill = env.step(action)

            if next_obs is None:
                print("⚠️ 当前帧状态缺失，丢弃动作，重新采样")
                obs = None
                continue

            total_reward += reward
            step_count += 1

            # 准备下一个状态
            next_obs = next_obs.unsqueeze(0)
            obs = next_obs

        print(f"✅ 测试完成，总步数: {step_count}, 总奖励: {total_reward:.2f}")
        with open("testing_log.txt", "a") as log_file:
            log_file.write(
                f"epoch {ep_num}: step:{step_count}, reward:{total_reward:.4f}, kill: {kill}\n")

        env.close()


if __name__ == "__main__":
    ppo_model_test(100)
