#!/usr/bin/env python3
"""Plot PPO training reward over environment steps from a run's Monitor logs.

Usage:
    uv run sim/scripts/plot_rewards.py --run-name run_walk_improved
    uv run sim/scripts/plot_rewards.py --run-name run_walk_improved --output runs/run_walk_improved/reward.png
    uv run sim/scripts/plot_rewards.py --run-name run_walk_improved --monitor-subdir eval_monitor --window 5
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))  # allow `import sim.*` when run as a standalone script

import numpy as np


def plot_reward_over_steps(
    run_dir: Path,
    monitor_subdir: str = "monitor",
    output: Path | None = None,
    window: int = 20,
) -> None:
    """Plot per-episode reward against cumulative environment steps for a training run.

    Reads every `*.monitor.csv` (stable-baselines3 Monitor format: a JSON comment
    line, a `r,l,t` header, then one row per episode) under `run_dir/monitor_subdir`
    -- one file per parallel environment -- maps each episode's reward to its
    cumulative step count within that file, merges all environments' points, and
    plots raw + rolling-mean reward vs steps.
    """
    monitor_dir = run_dir / monitor_subdir
    monitor_files = sorted(monitor_dir.glob("*.monitor.csv"))
    if not monitor_files:
        raise FileNotFoundError(f"no *.monitor.csv files found in {monitor_dir}")

    import matplotlib
    if output is not None:
        matplotlib.use("Agg")  # headless-safe backend, must be set before importing pyplot
    import matplotlib.pyplot as plt

    all_steps, all_rewards = [], []
    for monitor_file in monitor_files:
        data = np.loadtxt(monitor_file, delimiter=",", skiprows=2, ndmin=2)
        if data.size == 0:
            continue
        rewards, lengths = data[:, 0], data[:, 1]
        all_steps.append(np.cumsum(lengths))
        all_rewards.append(rewards)

    if not all_steps:
        raise ValueError(f"no episodes logged yet under {monitor_dir}")

    steps = np.concatenate(all_steps)
    rewards = np.concatenate(all_rewards)
    order = np.argsort(steps)
    steps, rewards = steps[order], rewards[order]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, rewards, alpha=0.3, label="episode reward")
    if window > 1 and len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(steps[window - 1:], smoothed, label=f"rolling mean ({window} episodes)")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("episode reward")
    ax.set_title(f"{run_dir.name}: reward over steps")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150)
        print(f"saved plot to {output}")
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, nargs="?", help="path to a run directory (contains monitor/)")
    parser.add_argument("--run-name", help="shortcut for runs/<run-name>")
    parser.add_argument("--monitor-subdir", default="monitor", help="monitor/ or eval_monitor/ (default: monitor)")
    parser.add_argument("--window", type=int, default=20, help="rolling-mean window in episodes (default: 20)")
    parser.add_argument("--output", type=Path, help="save the plot here instead of showing it interactively")
    args = parser.parse_args()

    if args.run_dir is None:
        if args.run_name is None:
            parser.error("pass a run_dir, or --run-name for runs/<run-name>")
        args.run_dir = REPO_ROOT / "runs" / args.run_name
    if not args.run_dir.exists():
        parser.error(f"no run directory at {args.run_dir}")

    plot_reward_over_steps(args.run_dir, args.monitor_subdir, args.output, args.window)


if __name__ == "__main__":
    main()
