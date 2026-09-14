#!/usr/bin/env python3
"""CLI: train a PPO walking policy for TonyPi.

Usage:
    uv run sim/scripts/train_ppo.py --config configs/ppo_flat.yaml
    uv run sim/scripts/train_ppo.py --config configs/ppo_flat.yaml --resume runs/ppo_flat/checkpoints/ppo_1000000_steps.zip
    uv run sim/scripts/train_ppo.py --config configs/ppo_flat.yaml --device cpu
"""
import argparse
import sys
from dataclasses import asdict, fields
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
    parser.add_argument("--device", default="auto", help="PyTorch device, e.g. cpu, cuda, or auto (default: auto)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.run_name:
        cfg.run_name = args.run_name

    run_dir = REPO_ROOT / "runs" / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)
    print(f"Training configuration saved to {config_path}")
    print(yaml.safe_dump(asdict(cfg), sort_keys=False), end="")

    final_path = train(cfg, runs_dir=REPO_ROOT / "runs", resume_from=args.resume, device=args.device)
    print(f"Saved final model to {final_path}")


if __name__ == "__main__":
    main()
