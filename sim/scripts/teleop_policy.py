#!/usr/bin/env python3
"""Interactively drive a trained TonyPi policy with the arrow keys.

Up/Down set the commanded forward/backward speed; Left/Right set the commanded
turn rate; Space stops the robot. The policy itself is unchanged -- this script
only updates TonyPiFlatEnv's velocity command live, via the MuJoCo viewer's
key callback, while the trained policy keeps producing joint actions to track it.

Usage:
    uv run sim/scripts/teleop_policy.py --run-name ppo_flat
    uv run sim/scripts/teleop_policy.py runs/ppo_flat/final_model.zip
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))  # allow `import sim.*` when run as a standalone script

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import sim.envs  # noqa: E402, F401 registers TonyPiFlat-v0 with gymnasium

# GLFW key codes, as used by mujoco.viewer's key_callback (avoids a separate glfw import).
KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_SPACE = 265, 264, 263, 262, 32
FORWARD_SPEED = 0.2
YAW_RATE = 0.6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path, nargs="?", help="checkpoint .zip (omit if using --run-name)")
    parser.add_argument("--run-name", help="load runs/<run-name>/best_model/best_model.zip (from EvalCallback)")
    parser.add_argument("--env-id", default="TonyPiFlat-v0")
    parser.add_argument("--device", default="auto", help="PyTorch device, e.g. cpu, cuda, or auto (default: auto)")
    args = parser.parse_args()

    if args.model_path is None:
        if args.run_name is None:
            parser.error("pass a model_path, or --run-name to load its best checkpoint")
        args.model_path = REPO_ROOT / "runs" / args.run_name / "best_model" / "best_model.zip"
    if not args.model_path.exists():
        parser.error(f"no checkpoint at {args.model_path}")

    env = gym.make(args.env_id, render_mode=None, randomize_command=False)
    model = PPO.load(args.model_path, device=args.device)

    # same VecNormalize-loading logic as eval_policy.py -- see its comment for why this matters.
    candidates = [args.model_path.parent / "vecnormalize.pkl", args.model_path.parent.parent / "vecnormalize.pkl"]
    vecnormalize_path = next((p for p in candidates if p.exists()), None)
    vec_env = None
    if vecnormalize_path is not None:
        vec_env = VecNormalize.load(str(vecnormalize_path), DummyVecEnv([lambda: env]))
        vec_env.training = False
        vec_env.norm_reward = False

    command = np.zeros(3)

    def on_key(keycode: int) -> None:
        if keycode == KEY_UP:
            command[0] = FORWARD_SPEED
        elif keycode == KEY_DOWN:
            command[0] = -FORWARD_SPEED
        elif keycode == KEY_LEFT:
            command[2] = YAW_RATE
        elif keycode == KEY_RIGHT:
            command[2] = -YAW_RATE
        elif keycode == KEY_SPACE:
            command[:] = 0.0
        else:
            return
        env.unwrapped.set_velocity_command(*command)

    obs = vec_env.reset() if vec_env is not None else env.reset()[0]
    print("Controls: Up/Down = forward/backward, Left/Right = turn, Space = stop")
    with mujoco.viewer.launch_passive(env.unwrapped.model, env.unwrapped.data, key_callback=on_key) as viewer:
        while viewer.is_running():
            action, _ = model.predict(obs, deterministic=True)
            if vec_env is not None:
                obs, _, done_arr, _ = vec_env.step(action)
                done = bool(done_arr[0])
            else:
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            viewer.sync()
            if done:
                obs = vec_env.reset() if vec_env is not None else env.reset()[0]
    env.close()


if __name__ == "__main__":
    main()
