# PPO training loop for TonyPi locomotion tasks, built on stable-baselines3.
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

import sim.envs  # noqa: F401 registers TonyPiFlat-v0 with gymnasium


@dataclass
class PPOConfig:
    env_id: str = "TonyPiFlat-v0"
    run_name: str = "ppo_flat"
    total_timesteps: int = 5_000_000
    n_envs: int = 8
    subproc: bool = True  # run envs in separate processes (set False if pickling/debugging issues)
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 10
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    checkpoint_freq: int = 200_000
    net_arch: list[int] = field(default_factory=lambda: [256, 256])
    seed: int | None = None
    normalize: bool = True  # VecNormalize obs/reward -- helps PPO a lot on MuJoCo continuous-control tasks
    eval_freq: int = 50_000
    n_eval_episodes: int = 5
    progress_bar: bool = True


def _make_env(cfg: PPOConfig, run_dir: Path, n_envs: int, monitor_subdir: str):
    vec_env_cls = SubprocVecEnv if (cfg.subproc and n_envs > 1) else DummyVecEnv
    vec_env = make_vec_env(
        cfg.env_id, n_envs=n_envs, seed=cfg.seed,
        monitor_dir=str(run_dir / monitor_subdir), vec_env_cls=vec_env_cls,
    )
    if cfg.normalize:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=(monitor_subdir == "monitor"))
    return vec_env


def train(
    cfg: PPOConfig,
    runs_dir: Path,
    resume_from: Path | None = None,
    device: str = "auto",
) -> Path:
    """Train (or resume) a PPO policy and return the path to the final saved model."""
    run_dir = runs_dir / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    vec_env = _make_env(cfg, run_dir, cfg.n_envs, "monitor")
    # eval_env always single-env; EvalCallback syncs its obs_rms from vec_env before every
    # evaluation, so its own normalize stats only need training=False (frozen, no reward norm).
    eval_env = _make_env(cfg, run_dir, 1, "eval_monitor")
    if cfg.normalize:
        eval_env.training = False

    vecnormalize_path = run_dir / "vecnormalize.pkl"
    if resume_from is not None and cfg.normalize and vecnormalize_path.exists():
        vec_env = VecNormalize.load(str(vecnormalize_path), vec_env.venv)

    if resume_from is not None:
        model = PPO.load(resume_from, env=vec_env, tensorboard_log=str(run_dir / "tb"), device=device)
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            learning_rate=cfg.learning_rate,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            clip_range=cfg.clip_range,
            ent_coef=cfg.ent_coef,
            policy_kwargs={"net_arch": cfg.net_arch},
            tensorboard_log=str(run_dir / "tb"),
            seed=cfg.seed,
            verbose=1,
            device=device,
        )

    callback = CallbackList([
        CheckpointCallback(
            save_freq=max(cfg.checkpoint_freq // cfg.n_envs, 1),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="ppo",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(run_dir / "best_model"),
            log_path=str(run_dir / "eval"),
            eval_freq=max(cfg.eval_freq // cfg.n_envs, 1),
            n_eval_episodes=cfg.n_eval_episodes,
            deterministic=True,
        ),
    ])

    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=callback,
        reset_num_timesteps=resume_from is None,
        tb_log_name=cfg.run_name,
        progress_bar=cfg.progress_bar,
    )

    final_path = run_dir / "final_model.zip"
    model.save(final_path)
    if cfg.normalize:
        vec_env.save(str(vecnormalize_path))
    return final_path
