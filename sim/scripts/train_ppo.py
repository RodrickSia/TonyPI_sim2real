#!/usr/bin/env python3
"""CLI: train a PPO walking policy for TonyPi.

Usage:
    uv run sim/scripts/train_ppo.py --config configs/ppo_flat.yaml
    uv run sim/scripts/train_ppo.py --config configs/ppo_flat.yaml --resume runs/ppo_flat/checkpoints/ppo_1000000_steps.zip
"""
import argparse
import sys
from dataclasses import fields
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))  # allow `import sim.*` when run as a standalone script

from sim.train.ppo import PPOConfig, train  # noqa: E402


def load_config(path: Path) -> PPOConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    valid_keys = {f.name for f in fields(PPOConfig)}
    unknown = set(raw) - valid_keys
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    return PPOConfig(**raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "ppo_flat.yaml")
    parser.add_argument("--resume", type=Path, default=None, help="checkpoint .zip to resume training from")
    parser.add_argument("--run-name", type=str, default=None, help="override run_name from the config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.run_name:
        cfg.run_name = args.run_name

    final_path = train(cfg, runs_dir=REPO_ROOT / "runs", resume_from=args.resume)
    print(f"Saved final model to {final_path}")


if __name__ == "__main__":
    main()
