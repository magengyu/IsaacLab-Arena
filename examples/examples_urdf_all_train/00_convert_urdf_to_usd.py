#!/usr/bin/env python3
"""Convert the tiny teaching URDF into an Isaac Lab friendly USD.

Run inside the IsaacLab-Arena Docker workspace:

    /isaac-sim/python.sh examples/examples_urdf_all_train/00_convert_urdf_to_usd.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaacsim import SimulationApp


def _set_if_present(obj: object, name: str, value: object) -> None:
    if hasattr(obj, name):
        setattr(obj, name, value)


def _move_articulation_root_to_default_prim(usd_path: Path, default_prim_name: str) -> None:
    root_text = usd_path.read_text(encoding="utf-8")
    root_text = root_text.replace(
        f'def Xform "{default_prim_name}" (\n'
        "    prepend references = @./payloads/base.usda@",
        f'def Xform "{default_prim_name}" (\n'
        '    prepend apiSchemas = ["PhysicsArticulationRootAPI", "NewtonArticulationRootAPI"]\n'
        "    prepend references = @./payloads/base.usda@",
    )
    usd_path.write_text(root_text, encoding="utf-8")

    physics_path = usd_path.parent / "payloads/Physics/physics.usda"
    if physics_path.exists():
        physics_text = physics_path.read_text(encoding="utf-8")
        physics_text = physics_text.replace(
            '    over "Geometry" (\n'
            '        prepend apiSchemas = ["PhysicsArticulationRootAPI", "NewtonArticulationRootAPI"]\n'
            "    )",
            '    over "Geometry"',
        )
        physics_path.write_text(physics_text, encoding="utf-8")


def _author_drive_overrides(stage, default_prim_name: str, stiffness: float, damping: float) -> None:
    from pxr import UsdPhysics

    for index in range(1, 7):
        joint_path = f"/{default_prim_name}/Physics/joint_{index}"
        joint_prim = stage.GetPrimAtPath(joint_path)
        if not joint_prim.IsValid():
            print(f"[WARN] joint not found for drive patch: {joint_path}")
            continue
        drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
        drive.CreateStiffnessAttr(stiffness)
        drive.CreateDampingAttr(damping)
        print(f"[INFO] drive joint_{index}: stiffness={stiffness:g}, damping={damping:g}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert simple_6dof_arm.urdf to USD.")
    parser.add_argument(
        "--urdf",
        type=Path,
        default=Path("examples/examples_urdf_all_train/assets/simple_urdf_arm/simple_6dof_arm.urdf"),
    )
    parser.add_argument(
        "--usd-output-dir",
        type=Path,
        default=Path("examples/examples_urdf_all_train/assets/simple_urdf_arm/usd"),
    )
    parser.add_argument("--stiffness", type=float, default=80.0, help="Drive stiffness authored into USD.")
    parser.add_argument("--damping", type=float, default=8.0, help="Drive damping authored into USD.")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    sim_app = SimulationApp({"headless": args.headless})
    try:
        import omni.kit.app
        import omni.usd
        from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig

        urdf_path = args.urdf.resolve()
        usd_output_dir = args.usd_output_dir.resolve()
        usd_output_dir.mkdir(parents=True, exist_ok=True)

        config = URDFImporterConfig()
        config.urdf_path = str(urdf_path)
        config.usd_path = str(usd_output_dir)
        config.make_default_prim = True
        config.fix_base = True
        config.merge_mesh = False
        config.collision_from_visuals = False
        config.import_inertia_tensor = True
        _set_if_present(config, "default_drive_type", "position")

        importer = URDFImporter(config)
        usd_path = Path(importer.import_urdf()).resolve()
        print(f"[INFO] imported USD: {usd_path}")

        omni.usd.get_context().open_stage(str(usd_path))
        app = omni.kit.app.get_app()
        for _ in range(20):
            app.update()

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError(f"Failed to open generated USD: {usd_path}")
        default_prim = stage.GetDefaultPrim()
        if not default_prim.IsValid():
            raise RuntimeError(f"Generated USD has no default prim: {usd_path}")
        print(f"[INFO] defaultPrim: {default_prim.GetPath()}")

        _author_drive_overrides(stage, default_prim.GetName(), args.stiffness, args.damping)
        stage.GetRootLayer().Save()
        _move_articulation_root_to_default_prim(usd_path, default_prim.GetName())
        print(f"[INFO] saved Isaac Lab friendly USD: {usd_path}")
        print(usd_path)
    finally:
        sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
