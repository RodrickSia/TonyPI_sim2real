#!/usr/bin/env python3
# Quick mujoco.viewer sanity-check for any MJCF file passed as an argument.
import argparse
from pathlib import Path

import mujoco
import mujoco.viewer

SIM_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = SIM_DIR / "description" / "mjcf" / "scenes" / "flat_ground.xml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, nargs="?", default=DEFAULT_MODEL,
                         help=f"path to an MJCF file (default: {DEFAULT_MODEL})")
    parser.add_argument("--keyframe", default="home",
                         help="keyframe name to reset to on load, if present (default: home)")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.model)) # type: ignore[attr-defined] 
    data = mujoco.MjData(model) # type: ignore[attr-defined] 

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, args.keyframe) # type: ignore[attr-defined] 
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id) # type: ignore[attr-defined] 
        mujoco.mj_forward(model, data) # type: ignore[attr-defined] 

    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()

