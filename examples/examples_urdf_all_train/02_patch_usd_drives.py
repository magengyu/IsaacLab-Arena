#!/usr/bin/env python3
"""Manually edit drive stiffness/damping in an already converted USD.

This is intentionally separate from the converter so you can practice changing
USD drive values without rerunning URDF import.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaacsim import SimulationApp


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch simple arm USD drive values.")
    parser.add_argument(
        "--usd",
        type=Path,
        default=Path("examples/examples_urdf_all_train/assets/simple_urdf_arm/usd/simple_6dof_arm/simple_6dof_arm.usda"),
    )
    parser.add_argument("--stiffness", type=float, default=120.0)
    parser.add_argument("--damping", type=float, default=12.0)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    sim_app = SimulationApp({"headless": args.headless})
    try:
        import omni.kit.app
        import omni.usd
        from pxr import UsdPhysics

        usd_path = args.usd.resolve()
        omni.usd.get_context().open_stage(str(usd_path))
        app = omni.kit.app.get_app()
        for _ in range(20):
            app.update()

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError(f"Failed to open USD: {usd_path}")
        default_prim = stage.GetDefaultPrim()
        for index in range(1, 7):
            joint_path = f"{default_prim.GetPath()}/Physics/joint_{index}"
            joint_prim = stage.GetPrimAtPath(joint_path)
            if not joint_prim.IsValid():
                print(f"[WARN] missing joint: {joint_path}")
                continue
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
            drive.CreateStiffnessAttr(args.stiffness)
            drive.CreateDampingAttr(args.damping)
            print(f"[INFO] patched {joint_path}: stiffness={args.stiffness:g}, damping={args.damping:g}")

        stage.GetRootLayer().Save()
        print(f"[INFO] saved: {usd_path}")
    finally:
        sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
