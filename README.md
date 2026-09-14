# TonyPi Sim2Real

Simulation and learning workspace for the [Hiwonder TonyPi](https://www.hiwonder.com/products/tonypi)
AI humanoid robot (18 DOF: 16 body bus servos + 2 head servos). Goal: train
policies in MuJoCo and transfer them to the real robot.

## Repository layout

```
assets/          CAD/reference source (Blender rig, mesh STLs, reference docs/media)
configs/         PPO/training hyperparameter configs (yaml)
control/         (planned) real-robot control / deployment code
data/            teleop logs and processed datasets
docs/            project documentation
infra/           (planned) infrastructure/tooling
models/          vla/ and world_model/ model code and checkpoints
runs/            (gitignored) PPO checkpoints + tensorboard logs, per run
sim/             MuJoCo simulation: robot description, envs, training, scripts
```

### `sim/` details

```
sim/description/
  urdf/TonyPi.urdf          source-of-truth kinematic tree (link/joint layout)
  mjcf/tonypi.xml           robot MJCF: converted from the URDF + hand-added
                            actuators/defaults (no floor/world in this file)
  mjcf/scenes/flat_ground.xml   flat-ground scene: includes tonypi.xml, adds
                            floor/lights/skybox and a "home" standing keyframe
  meshes/visual/           real STL meshes (from the rigged tonypi.glb rig); collision stays primitive
sim/envs/
  flat_env.py              Gymnasium env (TonyPiFlat-v0) wrapping flat_ground.xml
sim/train/
  ppo.py                   PPOConfig + train() — stable-baselines3 PPO training loop
sim/scripts/
  convert_urdf_to_mjcf.py  regenerates a raw MJCF dump from the URDF
  view_model.py             opens any MJCF file in the interactive MuJoCo viewer
  train_ppo.py             CLI: trains a PPO policy from a configs/*.yaml, saves to runs/
  eval_policy.py           CLI: rolls out a saved checkpoint in the interactive viewer
```

## Setup

Requires Python >= 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

```bash
# open the flat-ground scene in the interactive viewer
uv run sim/scripts/view_model.py

# use the RL environment
uv run python -c "
import gymnasium
import sim.envs
env = gymnasium.make('TonyPiFlat-v0')
obs, info = env.reset()
"

# train a PPO walking policy (checkpoints + tensorboard logs land in runs/<run_name>/)
uv run sim/scripts/train_ppo.py --config configs/ppo_flat.yaml
tensorboard --logdir runs

# watch a trained checkpoint in the interactive viewer
uv run sim/scripts/eval_policy.py runs/ppo_flat/final_model.zip
```

## Known caveats

- Joint/servo layout (18 DOF) is wiggle-test-confirmed against the real robot:
  3 DOF/arm (shoulder_pitch, upperarm_roll, elbow — no wrist) and 5 DOF/leg
  (hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll). Visual meshes are real
  STLs extracted from the rigged Blender model; collision stays primitive.
- Actuator gains (`kp`/`kv`) and joint damping/armature/friction are
  placeholders, not tuned to the real servos.
- `TonyPiFlat-v0` is a command-conditioned walking task. It observes a 45-value
  vector: 18 calibrated servo angles (radians), 18 finite-difference servo
  velocities (radians/s), the IMU values `ax, ay, az, gx, gy, gz`, and a 3-value
  velocity command (`vx`, `vy`, `yaw_rate`) in the robot's base frame. The
  command is randomized during training so the policy learns to track arbitrary
  requests, and can be set live via `env.unwrapped.set_velocity_command(...)`,
  e.g. from a keyboard/gamepad teleop loop (`sim/scripts/teleop_policy.py`).
  This mirrors the real robot's existing gamepad interface
  (`hiwonder.Board.get_gamepad`), unlike an absolute world-frame goal point,
  which the real robot cannot directly observe.
- The Hiwonder SDK reads each servo position as a hardware pulse, not radians.
  Deployment must apply a per-servo pulse-to-angle calibration, including the
  servo ID, zero offset, direction, and valid range. The upstream TonyPi source
  does not provide an anatomical mapping for its numbered `Servo1`-`Servo18`
  channels, so that calibration must be measured on the target robot.
- IMU noise, bias, latency, servo quantization, and contact sensing are now
  simulated: servo reads are quantized to a 0-1000 pulse over a 240 deg sweep,
  matching the real bus servo's reporting resolution (confirmed by
  `angle_l, angle_h = 0, 1000` in `external/TonyPi/HiwonderSDK/hiwonder/
  ros_robot_controller_sdk.py`). Servo and IMU reads are also delayed by
  `sensor_read_delay_steps` control steps (default 1) to emulate bus/read
  latency, and the IMU adds Gaussian noise plus a per-episode fixed bias
  (`imu_*_noise_std`, `imu_*_bias_std`). These noise/bias/latency magnitudes are
  reasonable placeholders, not measured from the real IMU chip's datasheet, and
  should be tuned once real sensor logs are available. Foot-contact sensing is
  still not simulated; the model has no physical foot-contact sensor equivalent.
