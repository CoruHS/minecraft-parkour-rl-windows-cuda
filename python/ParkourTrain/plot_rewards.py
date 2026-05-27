"""Plot CNN training/eval reward over time.

Reads the CSV logs written by train_cnn.py and saves PNGs to training_logs/.
Does not touch Minecraft and can be run any time, even while training (the
CSVs are appended live).

Usage:
    python3 python/ParkourTrain/plot_rewards.py
    python3 python/ParkourTrain/plot_rewards.py --window 100
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed; write files only
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "training_logs"
EPISODE_LOG_PATH = LOG_DIR / "episodes_cnn.csv"
EVAL_LOG_PATH = LOG_DIR / "eval_cnn.csv"


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def moving_average(values, window):
    if window <= 1 or len(values) < window:
        return [], []
    averaged = []
    running = sum(values[:window])
    averaged.append(running / window)
    for i in range(window, len(values)):
        running += values[i] - values[i - window]
        averaged.append(running / window)
    x = list(range(window - 1, len(values)))
    return x, averaged


def plot_training(rows, window, out_path):
    if not rows:
        print(f"No training rows in {EPISODE_LOG_PATH}; skipping training plot.")
        return
    episodes = [int(r["episode"]) for r in rows]
    rewards = [float(r["episode_reward"]) for r in rows]

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, color="#9ecae1", linewidth=0.8, label="episode reward")
    avg_x, avg_y = moving_average(rewards, window)
    if avg_y:
        # shift x to align with episode numbers
        avg_episodes = [episodes[i] for i in avg_x]
        plt.plot(avg_episodes, avg_y, color="#08519c", linewidth=2.0,
                 label=f"moving avg ({window})")
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.title("CNN training reward over time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved {out_path}")


def plot_eval(rows, out_path):
    if not rows:
        print(f"No eval rows in {EVAL_LOG_PATH}; skipping eval plot "
              "(run train_cnn.py until at least one eval happens).")
        return
    steps = [int(r["steps_done"]) for r in rows]
    mean = [float(r["mean_reward"]) for r in rows]
    worst = [float(r["worst_reward"]) for r in rows]
    best = [float(r["best_reward"]) for r in rows]
    entropy = [float(r["mean_entropy"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.fill_between(steps, worst, best, color="#c6dbef", alpha=0.6, label="worst..best")
    ax1.plot(steps, mean, color="#08519c", linewidth=2.0, marker="o", label="mean")
    ax1.set_ylabel("eval reward")
    ax1.set_title("CNN deterministic eval reward over time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # entropy tracks policy collapse: a steady drop means the greedy policy is
    # concentrating on a few actions.
    ax2.plot(steps, entropy, color="#d94801", linewidth=2.0, marker="o")
    ax2.set_xlabel("steps_done")
    ax2.set_ylabel("mean policy entropy")
    ax2.set_title("Eval policy entropy (lower = more concentrated)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot CNN reward over time.")
    parser.add_argument("--window", type=int, default=50,
                        help="Moving-average window for the training reward plot.")
    return parser.parse_args()


def main():
    args = parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    plot_training(read_csv(EPISODE_LOG_PATH), args.window, LOG_DIR / "reward_training.png")
    plot_eval(read_csv(EVAL_LOG_PATH), LOG_DIR / "reward_eval.png")


if __name__ == "__main__":
    main()
