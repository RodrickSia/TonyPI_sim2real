#!/usr/bin/env python3
# One-shot/rerunnable URDF -> MJCF compiler dump for TonyPi.urdf.
# Loads the URDF with MuJoCo's built-in URDF compiler and saves the resulting
# kinematic tree as MJCF. The output still needs manual augmentation
# (actuators, sensors, contact excludes, defaults, keyframes) — see
# description/mjcf/tonypi.xml for the hand-tuned result.
import argparse
from pathlib import Path

import mujoco

SIM_DIR = Path(__file__).resolve().parent.parent
DEFAULT_URDF = SIM_DIR / "description" / "urdf" / "TonyPi.urdf"
DEFAULT_OUTPUT = SIM_DIR / "description" / "mjcf" / "tonypi_raw.xml"


def convert(urdf_path: Path, output_path: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(urdf_path))
    mujoco.mj_saveLastXML(str(output_path), model)
    print(f"Compiled {urdf_path} -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    convert(args.urdf, args.output)


if __name__ == "__main__":
    main()
