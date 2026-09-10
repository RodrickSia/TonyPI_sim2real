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
- No IMU/joint sensors or contact excludes yet.
