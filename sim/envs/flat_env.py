# Gymnasium Env wrapping description/mjcf/scenes/flat_ground.xml.
# Command-conditioned walking task: an external operator (keyboard, or a
# gamepad on the real robot, see hiwonder.Board.get_gamepad) sets a desired
# base-frame velocity, and the policy is rewarded for tracking it.
from collections import deque
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
TARGET_PELVIS_HEIGHT = 0.275
PELVIS_BODY_ID = 2
SERVO_POSITION_COUNT = 18
IMU_AXIS_COUNT = 6
COMMAND_SIZE = 3
HARDWARE_OBSERVATION_SIZE = 2 * SERVO_POSITION_COUNT + IMU_AXIS_COUNT
OBSERVATION_SIZE = HARDWARE_OBSERVATION_SIZE + COMMAND_SIZE
# Hiwonder bus servos report position as a 0-1000 pulse over a 240 deg sweep --
# confirmed by external/TonyPi/HiwonderSDK/hiwonder/ros_robot_controller_sdk.py
# (`angle_l, angle_h = 0, 1000` passed to bus_servo_set_angle_limit).
BUS_SERVO_PULSE_RANGE = 1000
BUS_SERVO_SWEEP_RADIANS = np.deg2rad(240.0)
SERVO_QUANTIZATION_RAD = BUS_SERVO_SWEEP_RADIANS / BUS_SERVO_PULSE_RANGE
OBSERVATION_LAYOUT = (
    "servo_positions_rad[18]",
    "servo_velocities_rad_s[18]",
    "imu_acceleration_m_s2[3]",
    "imu_angular_velocity_rad_s[3]",
    "velocity_command_base[vx_m_s, vy_m_s, yaw_rate_rad_s]",
)
DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": PELVIS_BODY_ID,
    "distance": 2.5,
    "azimuth": 120.0,
    "elevation": -20.0,
    "lookat": np.array([0.0, 0.0, TARGET_PELVIS_HEIGHT]),
}


class TonyPiFlatEnv(MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 50}

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        frame_skip: int = 10,  # dt = 0.002 (model timestep) * 10 = 0.02s -> matches render_fps=50
        forward_velocity_reward_weight: float = 1.0,
        lateral_velocity_reward_weight: float = 0.5,
        yaw_rate_reward_weight: float = 0.5,
        command_vx_range: tuple[float, float] = (0.0, 0.3),
        command_vy_range: tuple[float, float] = (0.0, 0.0),
        command_yaw_rate_range: tuple[float, float] = (-0.3, 0.3),
        randomize_command: bool = True,
        command_resample_steps: int = 200,
        simulate_servo_quantization: bool = True,
        sensor_read_delay_steps: int = 1,
        imu_accel_noise_std: float = 0.05,
        imu_gyro_noise_std: float = 0.01,
        imu_accel_bias_std: float = 0.05,
        imu_gyro_bias_std: float = 0.01,
        ctrl_cost_weight: float = 0.0001,
        healthy_reward: float = 1.0,
        healthy_z_range: tuple = HEALTHY_Z_RANGE,
        reset_noise_scale: float = 0.01,
        render_mode: str | None = None,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self, model_path, frame_skip, forward_velocity_reward_weight, lateral_velocity_reward_weight,
            yaw_rate_reward_weight, command_vx_range, command_vy_range, command_yaw_rate_range,
            randomize_command, command_resample_steps, simulate_servo_quantization, sensor_read_delay_steps,
            imu_accel_noise_std, imu_gyro_noise_std, imu_accel_bias_std, imu_gyro_bias_std, ctrl_cost_weight,
            healthy_reward, healthy_z_range, reset_noise_scale, render_mode, **kwargs,
        )
        self._forward_velocity_reward_weight = forward_velocity_reward_weight
        self._lateral_velocity_reward_weight = lateral_velocity_reward_weight
        self._yaw_rate_reward_weight = yaw_rate_reward_weight
        self._command_vx_range = command_vx_range
        self._command_vy_range = command_vy_range
        self._command_yaw_rate_range = command_yaw_rate_range
        self._randomize_command = randomize_command
        self._command_resample_steps = command_resample_steps
        self._simulate_servo_quantization = simulate_servo_quantization
        self._sensor_read_delay_steps = max(sensor_read_delay_steps, 0)
        self._imu_accel_noise_std = imu_accel_noise_std
        self._imu_gyro_noise_std = imu_gyro_noise_std
        self._imu_accel_bias_std = imu_accel_bias_std
        self._imu_gyro_bias_std = imu_gyro_bias_std
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale

        # Hardware-equivalent proprioception plus the current velocity command.
        observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float64,
        )

        MujocoEnv.__init__(
            self, model_path, frame_skip, observation_space=observation_space,
            render_mode=render_mode, default_camera_config=DEFAULT_CAMERA_CONFIG, **kwargs,
        )

        # start from the "home" standing keyframe instead of MuJoCo's zero pose,
        # which has the robot half-buried in the floor
        home_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_id != -1:
            self.init_qpos = self.model.key_qpos[home_id].copy()

        self._pelvis_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self._base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self._previous_action = np.zeros(self.model.nu)
        actuator_joint_ids = self.model.actuator_trnid[:, 0]
        self._actuator_qpos_adrs = self.model.jnt_qposadr[actuator_joint_ids]
        self._actuator_joint_ranges = self.model.jnt_range[actuator_joint_ids]
        self._home_actuator_qpos = self.init_qpos[self._actuator_qpos_adrs].copy()
        self._imu_accelerometer_adr = self._sensor_data_address("imu_accelerometer")
        self._imu_gyroscope_adr = self._sensor_data_address("imu_gyroscope")
        self._velocity_command = np.zeros(3)  # vx, vy, yaw_rate in the base frame
        self._command_step_counter = 0
        self._imu_accel_bias = np.zeros(3)
        self._imu_gyro_bias = np.zeros(3)
        self._sensor_history_length = self._sensor_read_delay_steps + 1
        self._servo_position_history = deque(maxlen=self._sensor_history_length)
        self._imu_history = deque(maxlen=self._sensor_history_length)
        self._previous_servo_positions = self._quantize_servo_positions(self._servo_positions())
        self._reset_sensor_histories()

    def _sensor_data_address(self, sensor_name: str) -> int:
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        if sensor_id == -1:
            raise ValueError(f"Required sensor {sensor_name!r} is missing from the MuJoCo model")
        return self.model.sensor_adr[sensor_id]

    def _servo_positions(self) -> np.ndarray:
        return self.data.qpos[self._actuator_qpos_adrs].copy()

    # emulates the bus servo's 0-1000 pulse resolution (see SERVO_QUANTIZATION_RAD)
    def _quantize_servo_positions(self, positions: np.ndarray) -> np.ndarray:
        if not self._simulate_servo_quantization:
            return positions
        return np.round(positions / SERVO_QUANTIZATION_RAD) * SERVO_QUANTIZATION_RAD

    def _read_imu(self) -> np.ndarray:
        acceleration = self.data.sensordata[self._imu_accelerometer_adr:self._imu_accelerometer_adr + 3]
        angular_velocity = self.data.sensordata[self._imu_gyroscope_adr:self._imu_gyroscope_adr + 3]
        acceleration = acceleration + self._imu_accel_bias + self.np_random.normal(0.0, self._imu_accel_noise_std, size=3)
        angular_velocity = angular_velocity + self._imu_gyro_bias + self.np_random.normal(0.0, self._imu_gyro_noise_std, size=3)
        return np.concatenate([acceleration, angular_velocity])

    def _reset_sensor_histories(self) -> None:
        # fills the read-delay buffer so the first observation isn't a stale zero
        initial_servo_positions = self._quantize_servo_positions(self._servo_positions())
        initial_imu = self._read_imu()
        self._servo_position_history.clear()
        self._imu_history.clear()
        for _ in range(self._sensor_history_length):
            self._servo_position_history.append(initial_servo_positions)
            self._imu_history.append(initial_imu)

    # robot-frame (vx, vy): forward/lateral speed, as an operator would command it
    def _base_linear_velocity(self) -> np.ndarray:
        world_from_base = self.data.xmat[self._base_body_id].reshape(3, 3)[:2, :2]
        return world_from_base.T @ self.data.qvel[:2]

    def _sample_command(self) -> np.ndarray:
        vx = self.np_random.uniform(*self._command_vx_range)
        vy = self.np_random.uniform(*self._command_vy_range)
        yaw_rate = self.np_random.uniform(*self._command_yaw_rate_range)
        return np.array([vx, vy, yaw_rate])

    # externally set the commanded velocity, e.g. from a keyboard/gamepad teleop loop
    def set_velocity_command(self, forward: float, lateral: float = 0.0, yaw_rate: float = 0.0) -> None:
        self._velocity_command = np.array([forward, lateral, yaw_rate])

    def _get_obs(self):
        self._servo_position_history.append(self._quantize_servo_positions(self._servo_positions()))
        self._imu_history.append(self._read_imu())
        # oldest buffered sample == the simulated read delay's worth of latency
        servo_positions = self._servo_position_history[0]
        imu_acceleration, imu_angular_velocity = np.split(self._imu_history[0], 2)
        servo_velocities = (servo_positions - self._previous_servo_positions) / self.dt
        self._previous_servo_positions = servo_positions.copy()
        return np.concatenate([
            servo_positions,
            servo_velocities,
            imu_acceleration,
            imu_angular_velocity,
            self._velocity_command,
        ])

    @property
    def is_healthy(self) -> bool:
        min_z, max_z = self._healthy_z_range
        pelvis_z = self.data.xpos[self._pelvis_body_id, 2]
        pelvis_up = self.data.xmat[self._pelvis_body_id].reshape(3, 3)[:, 2]
        return min_z < pelvis_z < max_z and pelvis_up[2] > np.cos(MAX_TILT_RADIANS)

    def step(self, action):
        # simulate an operator occasionally issuing a new command mid-episode
        self._command_step_counter += 1
        if (
            self._randomize_command
            and self._command_resample_steps > 0
            and self._command_step_counter % self._command_resample_steps == 0
        ):
            self._velocity_command = self._sample_command()

        self.do_simulation(action, self.frame_skip)

        base_velocity = self._base_linear_velocity()
        yaw_rate = self.data.qvel[5]
        roll_pitch_rate = self.data.qvel[3:5]
        pelvis_up = self.data.xmat[self._pelvis_body_id].reshape(3, 3)[:, 2]
        pelvis_z = self.data.xpos[self._pelvis_body_id, 2]
        actuator_qpos = self.data.qpos[self._actuator_qpos_adrs]
        joint_margin = np.minimum(actuator_qpos - self._actuator_joint_ranges[:, 0], self._actuator_joint_ranges[:, 1] - actuator_qpos)
        forward_velocity_reward = self._forward_velocity_reward_weight * np.exp(-16.0 * np.square(base_velocity[0] - self._velocity_command[0]))
        lateral_velocity_reward = self._lateral_velocity_reward_weight * np.exp(-16.0 * np.square(base_velocity[1] - self._velocity_command[1]))
        yaw_rate_reward = self._yaw_rate_reward_weight * np.exp(-4.0 * np.square(yaw_rate - self._velocity_command[2]))
        upright_reward = np.square(pelvis_up[2])
        height_reward = np.exp(-200.0 * np.square(pelvis_z - TARGET_PELVIS_HEIGHT))
        angular_cost = 0.05 * np.sum(np.square(roll_pitch_rate))  # yaw rate is commanded, not penalized here
        ctrl_cost = self._ctrl_cost_weight * np.sum(np.square(self.data.actuator_force))
        action_rate_cost = 0.01 * np.sum(np.square(action - self._previous_action))
        posture_cost = 0.02 * np.sum(np.square(actuator_qpos[:8] - self._home_actuator_qpos[:8]))
        joint_limit_cost = 0.1 * np.sum(np.square(np.clip(0.15 - joint_margin, 0.0, None)))
        healthy_reward = self._healthy_reward if self.is_healthy else 0.0
        reward = forward_velocity_reward + lateral_velocity_reward + yaw_rate_reward + upright_reward + height_reward + healthy_reward - angular_cost - ctrl_cost - action_rate_cost - posture_cost - joint_limit_cost
        self._previous_action = action.copy()

        terminated = not self.is_healthy
        observation = self._get_obs()
        info = {
            "forward_velocity_reward": forward_velocity_reward,
            "lateral_velocity_reward": lateral_velocity_reward,
            "yaw_rate_reward": yaw_rate_reward,
            "upright_reward": upright_reward,
            "height_reward": height_reward,
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
        if self._randomize_command:
            self._velocity_command = self._sample_command()
        self._command_step_counter = 0
        self._imu_accel_bias = self.np_random.normal(0.0, self._imu_accel_bias_std, size=3)
        self._imu_gyro_bias = self.np_random.normal(0.0, self._imu_gyro_bias_std, size=3)
        self._previous_action.fill(0.0)
        self._reset_sensor_histories()
        self._previous_servo_positions = self._servo_position_history[0].copy()
        return self._get_obs()
