# Gymnasium Env wrapping description/mjcf/scenes/flat_ground.xml.
# Basic standing/forward-walking task, no custom sensors — observations are
# read directly from qpos/qvel via MuJoCo's built-in state arrays.
from pathlib import Path

import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_MODEL_PATH = str(
    Path(__file__).resolve().parent.parent / "description" / "mjcf" / "scenes" / "flat_ground.xml"
)
HEALTHY_Z_RANGE = (0.12, 0.35)  # base height bounds outside which the episode ends


class TonyPiFlatEnv(MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 50}

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        frame_skip: int = 10,  # dt = 0.002 (model timestep) * 10 = 0.02s -> matches render_fps=50
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 0.05,
        healthy_reward: float = 1.0,
        healthy_z_range: tuple = HEALTHY_Z_RANGE,
        reset_noise_scale: float = 0.01,
        render_mode: str | None = None,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self, model_path, frame_skip, forward_reward_weight, ctrl_cost_weight,
            healthy_reward, healthy_z_range, reset_noise_scale, render_mode, **kwargs,
        )
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale

        # qpos (25) minus base x,y + qvel (24) -> 47-dim observation
        observation_space = Box(low=-np.inf, high=np.inf, shape=(47,), dtype=np.float64)

        MujocoEnv.__init__(
            self, model_path, frame_skip, observation_space=observation_space,
            render_mode=render_mode, **kwargs,
        )

        # start from the "home" standing keyframe instead of MuJoCo's zero pose,
        # which has the robot half-buried in the floor
        home_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_id != -1:
            self.init_qpos = self.model.key_qpos[home_id].copy()

    def _get_obs(self):
        position = self.data.qpos.flatten()[2:]  # drop x,y: translation-invariant obs
        velocity = self.data.qvel.flatten()
        return np.concatenate([position, velocity])

    @property
    def is_healthy(self) -> bool:
        min_z, max_z = self._healthy_z_range
        return min_z < self.data.qpos[2] < max_z

    def step(self, action):
        x_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_after = self.data.qpos[0]

        forward_reward = self._forward_reward_weight * (x_after - x_before) / self.dt
        ctrl_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        healthy_reward = self._healthy_reward if self.is_healthy else 0.0
        reward = forward_reward + healthy_reward - ctrl_cost

        terminated = not self.is_healthy
        observation = self._get_obs()
        info = {
            "forward_reward": forward_reward,
            "ctrl_cost": ctrl_cost,
            "healthy_reward": healthy_reward,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def reset_model(self):
        noise_low, noise_high = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(noise_low, noise_high, self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()
