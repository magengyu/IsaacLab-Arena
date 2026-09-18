#!/usr/bin/env python3
"""Inspect links, joints, limits, axes, and drive gains in the converted USD."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaacsim import SimulationApp


def _rel_targets(prim, rel_name: str) -> list[str]:
    rel = prim.GetRelationship(rel_name)
    if not rel:
        return []
    return [str(path) for path in rel.GetTargets()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a simple arm USD.")
    parser.add_argument(
        "--usd",
        type=Path,
        default=Path("examples/examples_urdf_all_train/assets/simple_urdf_arm/usd/simple_6dof_arm/simple_6dof_arm.usda"),
    )
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
        print(f"[INFO] USD: {usd_path}")
        print(f"[INFO] defaultPrim: {default_prim.GetPath()}")
        print(f"[INFO] defaultPrim APIs: {default_prim.GetAppliedSchemas()}")

        print("\n[LINKS]")
        for prim in stage.Traverse():
            if "PhysicsRigidBodyAPI" in prim.GetAppliedSchemas():
                print(f"  {prim.GetPath()}")

        print("\n[JOINTS]")
        for prim in stage.Traverse():
            if prim.GetTypeName() not in {"PhysicsRevoluteJoint", "PhysicsPrismaticJoint", "PhysicsFixedJoint"}:
                continue
            print(f"  {prim.GetPath()} ({prim.GetTypeName()})")
            print(f"    body0: {_rel_targets(prim, 'physics:body0')}")
            print(f"    body1: {_rel_targets(prim, 'physics:body1')}")
            for attr_name in ("physics:axis", "physics:lowerLimit", "physics:upperLimit"):
                attr = prim.GetAttribute(attr_name)
                if attr:
                    print(f"    {attr_name}: {attr.Get()}")
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            if drive:
                print(f"    drive:angular:stiffness: {drive.GetStiffnessAttr().Get()}")
                print(f"    drive:angular:damping: {drive.GetDampingAttr().Get()}")
    finally:
        sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
