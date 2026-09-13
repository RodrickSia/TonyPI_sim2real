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
# pelvis (not base_link) height bounds outside which the episode ends: base_link
# sits at floor level in this model (pelvis is +0.267m above it), so it isn't a
# useful upright/fallen signal on its own. Standing pelvis height is ~0.275m.
HEALTHY_Z_RANGE = (0.15, 0.35)
MAX_TILT_RADIANS = 0.75
TARGET_FORWARD_VELOCITY = 0.25
TARGET_PELVIS_HEIGHT = 0.275


class TonyPiFlatEnv(MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 50}

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        frame_skip: int = 10,  # dt = 0.002 (model timestep) * 10 = 0.02s -> matches render_fps=50
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 0.0001,
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

        self._pelvis_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self._previous_action = np.zeros(self.model.nu)
        actuator_joint_ids = self.model.actuator_trnid[:, 0]
        self._actuator_qpos_adrs = self.model.jnt_qposadr[actuator_joint_ids]
        self._actuator_joint_ranges = self.model.jnt_range[actuator_joint_ids]
        self._home_actuator_qpos = self.init_qpos[self._actuator_qpos_adrs].copy()

    def _get_obs(self):
        position = self.data.qpos.flatten()[2:]  # drop x,y: translation-invariant obs
        velocity = self.data.qvel.flatten()
        return np.concatenate([position, velocity])

    @property
    def is_healthy(self) -> bool:
        min_z, max_z = self._healthy_z_range
        pelvis_z = self.data.xpos[self._pelvis_body_id, 2]
        pelvis_up = self.data.xmat[self._pelvis_body_id].reshape(3, 3)[:, 2]
        return min_z < pelvis_z < max_z and pelvis_up[2] > np.cos(MAX_TILT_RADIANS)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        velocity = self.data.qvel[:3]
        angular_velocity = self.data.qvel[3:6]
        pelvis_up = self.data.xmat[self._pelvis_body_id].reshape(3, 3)[:, 2]
        pelvis_z = self.data.xpos[self._pelvis_body_id, 2]
        actuator_qpos = self.data.qpos[self._actuator_qpos_adrs]
        joint_margin = np.minimum(actuator_qpos - self._actuator_joint_ranges[:, 0], self._actuator_joint_ranges[:, 1] - actuator_qpos)
        forward_reward = self._forward_reward_weight * np.exp(-16.0 * np.square(velocity[0] - TARGET_FORWARD_VELOCITY))
        upright_reward = np.square(pelvis_up[2])
        height_reward = np.exp(-200.0 * np.square(pelvis_z - TARGET_PELVIS_HEIGHT))
        lateral_cost = 0.5 * np.square(velocity[1])
        angular_cost = 0.05 * np.sum(np.square(angular_velocity))
        ctrl_cost = self._ctrl_cost_weight * np.sum(np.square(self.data.actuator_force))
        action_rate_cost = 0.01 * np.sum(np.square(action - self._previous_action))
        posture_cost = 0.02 * np.sum(np.square(actuator_qpos[:8] - self._home_actuator_qpos[:8]))
        joint_limit_cost = 0.1 * np.sum(np.square(np.clip(0.15 - joint_margin, 0.0, None)))
        healthy_reward = self._healthy_reward if self.is_healthy else 0.0
        reward = forward_reward + upright_reward + height_reward + healthy_reward - lateral_cost - angular_cost - ctrl_cost - action_rate_cost - posture_cost - joint_limit_cost
        self._previous_action = action.copy()

        terminated = not self.is_healthy
        observation = self._get_obs()
        info = {
            "forward_reward": forward_reward,
            "upright_reward": upright_reward,
            "height_reward": height_reward,
            "lateral_cost": lateral_cost,
            "angular_cost": angular_cost,
            "ctrl_cost": ctrl_cost,
            "action_rate_cost": action_rate_cost,
            "posture_cost": posture_cost,
            "joint_limit_cost": joint_limit_cost,
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
        self._previous_action.fill(0.0)
        return self._get_obs()
