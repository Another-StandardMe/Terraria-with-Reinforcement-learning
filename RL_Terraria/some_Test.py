def get_a_opt(step, C=200000):
    return (step**2) / (step**2 + C)

def get_a(step, beta_const=200):
    return step / (step + 2000 * (beta_const / step))



steps = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1200, 1400, 2000]

for i in steps:
    a = get_a(i)
    a_opt = get_a_opt(i)
    reward = (1 - a) * 600
    reward_opt = (1 - a_opt) * 600

    print(f"Step {i}:")
    print(f"  get_a     折扣比例: {a:.6f}, (1-a):{1-a}, 奖励: {reward:.2f}")
    print(f"  get_a_opt 折扣比例: {a_opt:.6f}, (1-a_opt):{1-a_opt}, 奖励: {reward_opt:.2f}")
    print("-" * 50)


import torch

class DummyRedistributor:
    def __init__(self, terminal_bonus_value=10):
        self.terminal_bonus_value = terminal_bonus_value

    def redistribute_final_reward(self, rewards, final_reward):
        eps = 3
        r_tensor = torch.stack(rewards).squeeze()
        r_main = r_tensor[:-1]  # 前 T 个状态
        T = len(r_main)

        if final_reward >= 0:
            min_r = r_main.min()
            weights = torch.where(r_main > 0, r_main - min_r + eps, torch.full_like(r_main, eps))
            print(f"weights：{weights}")
        else:
            max_r = r_main.max()
            weights = torch.where(r_main < 0, -r_main + max_r + eps, torch.full_like(r_main, eps))
            print(f"weights：{weights}")
        weights = weights / (weights.sum() + eps)
        print(f"weights：{weights}")
        redistribution = weights * final_reward
        adjusted = r_main + redistribution

        terminal_bonus = torch.tensor(
            self.terminal_bonus_value if final_reward >= 0 else -self.terminal_bonus_value,
            device=r_tensor.device
        )
        adjusted_rewards = list(adjusted) + [terminal_bonus]

        return [r.unsqueeze(0) for r in adjusted_rewards]


# 测试函数
def redistribute():
    redistributor = DummyRedistributor(terminal_bonus_value=10)

    test_cases = [
        {
            "name": "正向 final_reward",
            "input_rewards": [1, 1, -1, -1, -9, 1, 1, 5],
        },
        {
            "name": "负向 final_reward",
            "input_rewards": [1, 1, -1, -1, -9, 1, 1, -5],
        },
    ]

    for case in test_cases:
        print(f"\n==== {case['name']} ====")
        raw_rewards = case["input_rewards"]
        rewards = [torch.tensor([r], dtype=torch.float32) for r in raw_rewards]
        final_reward = raw_rewards[-1]

        adjusted = redistributor.redistribute_final_reward(rewards, final_reward)
        adjusted_flat = torch.cat(adjusted).cpu().numpy()

        print(f"原始 rewards:   {raw_rewards}")
        print(f"调整后 rewards: {adjusted_flat.tolist()}")

        total_reward = sum(adjusted_flat)
        print(f" 总奖励: {total_reward:.4f}  (应为 final_reward + terminal_bonus = {final_reward + (50 if final_reward > 0 else -50)})")


redistribute()
