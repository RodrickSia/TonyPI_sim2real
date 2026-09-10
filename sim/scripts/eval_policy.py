#!/usr/bin/env python3
"""CLI: roll out a trained PPO checkpoint in the interactive MuJoCo viewer.

Usage:
    uv run sim/scripts/eval_policy.py runs/ppo_flat/final_model.zip
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))  # allow `import sim.*` when run as a standalone script

import gymnasium as gym
from stable_baselines3 import PPO

import sim.envs  # noqa: E402, F401 registers TonyPiFlat-v0 with gymnasium


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--env-id", default="TonyPiFlat-v0")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    env = gym.make(args.env_id, render_mode="human")
    model = PPO.load(args.model_path)

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
        print(f"episode {ep}: reward={ep_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()
