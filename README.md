# TonyPi Sim2Real

Simulation and learning workspace for the [Hiwonder TonyPi](https://www.hiwonder.com/products/tonypi)
AI humanoid robot (18 DOF: 16 body bus servos + 2 head servos). Goal: train
policies in MuJoCo and transfer them to the real robot.

## Repository layout

```
assets/          CAD/reference source (Blender rig, reference docs/media)
control/         (planned) real-robot control / deployment code
data/            teleop logs and processed datasets
docs/            project documentation
infra/           (planned) infrastructure/tooling
models/          vla/ and world_model/ model code and checkpoints
sim/             MuJoCo simulation: robot description, envs, scripts
```

### `sim/` details

```
sim/description/
  urdf/TonyPi.urdf          source-of-truth kinematic tree (link/joint layout)
  mjcf/tonypi.xml           robot MJCF: converted from the URDF + hand-added
                            actuators/defaults (no floor/world in this file)
  mjcf/scenes/flat_ground.xml   flat-ground scene: includes tonypi.xml, adds
                            floor/lights/skybox and a "home" standing keyframe
  meshes/                  visual/collision meshes (placeholder boxes for now)
sim/envs/
  flat_env.py              Gymnasium env (TonyPiFlat-v0) wrapping flat_ground.xml
sim/scripts/
  convert_urdf_to_mjcf.py  regenerates a raw MJCF dump from the URDF
  view_model.py             opens any MJCF file in the interactive MuJoCo viewer
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
```

## Known caveats

- Link geometry is placeholder primitives (`<box>`), not real meshes.
- Joint/servo layout (18 DOF) is confirmed against the real robot, but exact
  per-servo IDs and the wrist's rotation axis are estimates pending further
  verification.
- Actuator gains (`kp`/`kv`) and joint damping/armature/friction are
  placeholders, not tuned to the real servos.
- No IMU/joint sensors or contact excludes yet.
