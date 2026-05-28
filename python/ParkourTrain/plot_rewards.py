"""Plot CNN training/eval reward over time.

Reads the CSV logs written by train_cnn.py and saves PNGs next to them.
Does not touch Minecraft and can be run any time, even while training (the
CSVs are appended live).

Each training run writes to its own training_logs/run_<timestamp>/ directory.
By default this script plots the newest such run; use --run to pick another.

Usage:
    python3 python/ParkourTrain/plot_rewards.py
    python3 python/ParkourTrain/plot_rewards.py --window 100
    python3 python/ParkourTrain/plot_rewards.py --run run_20260528_143012
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed; write files only
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "training_logs"


def find_latest_run_dir():
    """Return the newest training_logs/run_* directory, or None if there isn't one.

    Falls back to LOG_DIR itself if pre-run-id CSVs exist there (backward compat).
    """
    if not LOG_DIR.exists():
        return None
    candidates = sorted(
        (p for p in LOG_DIR.glob("run_*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
    )
    if candidates:
        return candidates[-1]
    legacy_csv = LOG_DIR / "episodes_cnn.csv"
    return LOG_DIR if legacy_csv.exists() else None


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


def plot_training(rows, window, out_path, source_path):
    if not rows:
        print(f"No training rows in {source_path}; skipping training plot.")
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


def plot_eval(rows, out_path, source_path):
    if not rows:
        print(f"No eval rows in {source_path}; skipping eval plot "
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
    parser.add_argument("--run", type=str, default=None,
                        help="Run directory to plot (e.g. 'run_20260528_143012'). "
                             "Defaults to the newest run under training_logs/.")
    return parser.parse_args()


def resolve_run_dir(run_arg):
    if run_arg is None:
        run_dir = find_latest_run_dir()
        if run_dir is None:
            raise FileNotFoundError(
                f"No run directory or legacy CSV found under {LOG_DIR}. "
                "Run train_cnn.py first or pass --run."
            )
        return run_dir
    candidate = Path(run_arg)
    if not candidate.is_absolute():
        candidate = LOG_DIR / candidate
    if not candidate.is_dir():
        raise FileNotFoundError(f"Run directory not found: {candidate}")
    return candidate


def main():
    args = parse_args()
    run_dir = resolve_run_dir(args.run)
    print(f"Plotting run: {run_dir}")
    episode_csv = run_dir / "episodes_cnn.csv"
    eval_csv = run_dir / "eval_cnn.csv"
    plot_training(read_csv(episode_csv), args.window, run_dir / "reward_training.png", episode_csv)
    plot_eval(read_csv(eval_csv), run_dir / "reward_eval.png", eval_csv)


if __name__ == "__main__":
    main()
