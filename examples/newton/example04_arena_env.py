# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""在 Arena 中使用 Newton 物理仿真的示例。

这个脚本展示如何用 Arena 的环境构建器组合一个包含桌面和物体的场景，并将物理
后端切换到 Newton。它适合快速验证 Isaac Lab Arena 与 Newton 的整合是否可用。

使用方法：

    .venv/bin/python examples/newton/example04_arena_env.py --num_steps 200 --keep_open
    .venv/bin/python examples/newton/example04_arena_env.py --viz kit --num_steps 200
    .venv/bin/python examples/newton/example04_arena_env.py --viz kit,newton --num_steps 200
    .venv/bin/python examples/newton/example04_arena_env.py --viz none --num_steps 50

注意：Isaac Lab 的 `AppLauncher` 只会在选择了 `kit` 视口时打开窗口；单独使用
`--viz newton` 会被当作 headless 模式，因此不会弹出可视化窗口。
"""

import argparse
import contextlib
import math
import time
from typing import Any, cast

import torch
from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser


def parse_args() -> argparse.Namespace:
    """Parse the demo CLI arguments."""
    parser = get_isaaclab_arena_cli_parser()
    parser.add_argument(
        "--num_steps",
        type=int,
        default=200,
        help="仿真步数。",
    )
    parser.add_argument(
        "--keep_open",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="仿真结束后是否保持可视化窗口打开。",
    )
    parser.set_defaults(
        num_envs=1,
        presets="newton",
        visualizer=["kit"],
        device="cuda:0",
    )
    return parser.parse_args()


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    """Create a Newton-safe Arena scene and step it with zero actions."""
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.embodiments.no_embodiment import NoEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose, PoseRange

    asset_registry = AssetRegistry()
    background = cast(Any, asset_registry.get_asset_by_name("procedural_table"))()
    background.set_initial_pose(Pose(position_xyz=(-0.55, 0.0, 0.235)))
    manip_object = cast(Any, asset_registry.get_asset_by_name("cracker_box"))()
    manip_object.set_initial_pose(
        PoseRange(
            position_xyz_min=(-0.75, -0.1, 0.35),
            position_xyz_max=(-0.35, 0.3, 0.75),
            rpy_min=(-math.pi, -math.pi, -math.pi),
            rpy_max=(math.pi, math.pi, math.pi),
        )
    )
    ground_plane = cast(Any, asset_registry.get_asset_by_name("ground_plane"))()
    light = cast(Any, asset_registry.get_asset_by_name("light"))()

    scene = Scene(assets=[background, manip_object, ground_plane, light])
    arena_env = IsaacLabArenaEnvironment(
        name="newton_arena_demo",
        embodiment=NoEmbodiment(),
        scene=scene,
    )

    env_builder = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli))
    env = env_builder.make_registered()
    env.reset()
    simulation_app.update()

    print(f"[INFO] Newton physics preset active: {args_cli.presets}", flush=True)
    print("[INFO] 开始 Newton 仿真循环。", flush=True)

    runtime_env = cast(Any, env)
    action = torch.zeros(runtime_env.action_space.shape, dtype=torch.float32, device=runtime_env.unwrapped.device)
    try:
        for step in range(args_cli.num_steps):
            # `simulation_app.is_running()` may still be false during Kit startup, so
            # a fixed-step loop is more reliable than breaking immediately on startup.
            with torch.inference_mode():
                runtime_env.step(action)
            simulation_app.update()
            time.sleep(runtime_env.unwrapped.step_dt)
            if step < 5 or step % 20 == 0:
                print(f"[INFO] step {step + 1}/{args_cli.num_steps}", flush=True)

        if args_cli.keep_open:
            print("[INFO] 保持窗口打开，关闭可视化窗口或按 Ctrl+C 退出。", flush=True)
            while simulation_app.is_running():
                simulation_app.update()
                time.sleep(1.0 / 60.0)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        with contextlib.suppress(KeyboardInterrupt):
            main()
    finally:
        simulation_app.close()
