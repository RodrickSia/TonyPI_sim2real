#!/usr/bin/env python3
"""CLI: roll out a trained PPO checkpoint, either in the interactive MuJoCo
viewer or (with --output) recorded to a video file -- use the latter on a
headless machine/Docker container with no display.

Usage:
    uv run sim/scripts/eval_policy.py runs/ppo_flat/final_model.zip
    uv run sim/scripts/eval_policy.py --run-name ppo_flat   # shortcut for the best checkpoint
    uv run sim/scripts/eval_policy.py --run-name ppo_flat --output runs/ppo_flat/eval.mp4 --device cpu
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))  # allow `import sim.*` when run as a standalone script

import gymnasium as gym
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import sim.envs  # noqa: E402, F401 registers TonyPiFlat-v0 with gymnasium


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path, nargs="?", help="checkpoint .zip (omit if using --run-name)")
    parser.add_argument("--run-name", help="load runs/<run-name>/best_model/best_model.zip (from EvalCallback)")
    parser.add_argument("--env-id", default="TonyPiFlat-v0")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--output", type=Path, help="record an mp4 here instead of opening the live viewer")
    parser.add_argument("--device", default="auto", help="PyTorch device, e.g. cpu, cuda, or auto (default: auto)")
    args = parser.parse_args()

    if args.model_path is None:
        if args.run_name is None:
            parser.error("pass a model_path, or --run-name to load its best checkpoint")
        args.model_path = REPO_ROOT / "runs" / args.run_name / "best_model" / "best_model.zip"
    if not args.model_path.exists():
        parser.error(f"no checkpoint at {args.model_path}")

    env = gym.make(args.env_id, render_mode="rgb_array" if args.output else "human")
    model = PPO.load(args.model_path, device=args.device)

    # if the run saved VecNormalize stats (sim/train/ppo.py does, when normalize=true),
    # apply the same obs normalization at inference -- otherwise the policy sees
    # out-of-distribution observations and behaves poorly. Stats live in the run
    # dir itself, so check both model_path's own dir (final_model.zip) and its
    # parent's parent (best_model/best_model.zip, checkpoints/ppo_*_steps.zip).
    candidates = [args.model_path.parent / "vecnormalize.pkl", args.model_path.parent.parent / "vecnormalize.pkl"]
    vecnormalize_path = next((p for p in candidates if p.exists()), None)
    vec_env = None
    if vecnormalize_path is not None:
        vec_env = VecNormalize.load(str(vecnormalize_path), DummyVecEnv([lambda: env]))
        vec_env.training = False
        vec_env.norm_reward = False

    writer = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(args.output, fps=env.unwrapped.metadata.get("render_fps", 50))

    for ep in range(args.episodes):
        obs, _ = (vec_env.reset(), None) if vec_env is not None else env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if vec_env is not None:
                obs, reward, done_arr, _ = vec_env.step(action)
                done = bool(done_arr[0])
                reward = reward[0]
            else:
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            ep_reward += reward
            if writer is not None:
                writer.append_data(env.render())
        print(f"episode {ep}: reward={ep_reward:.2f}")

    if writer is not None:
        writer.close()
        print(f"saved video to {args.output}")
    env.close()


if __name__ == "__main__":
    main()
