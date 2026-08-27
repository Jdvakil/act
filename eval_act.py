import sys
sys.path.append("/home/qinzhengfangli/molmo_test/molmospaces")

from molmo_spaces.evaluation.eval_main import run_evaluation, create_eval_config


def main():
    config = create_eval_config(
        num_episodes=10,   # 👉 Task 1 要求
        render=False       # 先关掉，快一点
    )

    results = run_evaluation(config)

    # 计算 success rate
    success = sum(r.success for r in results)
    total = len(results)

    print("\n🔥 SUCCESS RATE:", success / total)


if __name__ == "__main__":
    main()

