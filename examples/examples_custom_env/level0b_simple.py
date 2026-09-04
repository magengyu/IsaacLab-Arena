"""Level 0b: 简单 — 桌子 + 机器人 + 方块，几秒开"""
from dataclasses import dataclass
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.environments.arena_environment_factory import (
    ArenaEnvironmentCfg, ArenaEnvironmentFactory,
)
from isaaclab_arena.environments.isaaclab_arena_environment import (
    IsaacLabArenaEnvironment,
)
from isaaclab_arena.scene.scene import Scene


@dataclass
class Level0bCfg(ArenaEnvironmentCfg):
    robot: str = "franka_ik"
    object: str = "dex_cube"


class Level0bSimple(ArenaEnvironmentFactory[Level0bCfg]):
    name = "level0b_simple"
    _legacy_argparse_cfg_type = Level0bCfg

    def build(self, cfg: Level0bCfg) -> IsaacLabArenaEnvironment:
        # 桌子是背景（自带地面），不需要单独的 ground_plane
        table = self.asset_registry.get_asset_by_name("table")()
        robot = self.asset_registry.get_asset_by_name(cfg.robot)()
        obj = self.asset_registry.get_asset_by_name(cfg.object)()
        light = self.asset_registry.get_asset_by_name("light")()

        # 机械臂放在桌子左边，方块放在桌面上
        robot.set_initial_pose(Pose(position_xyz=(-0.4, 0.0, 0.0)))
        obj.set_initial_pose(Pose(position_xyz=(0.1, 0.0, 0.05)))
        scene = Scene(assets=[table, light, obj])

        return IsaacLabArenaEnvironment(
            name=self.name, embodiment=robot, scene=scene,
        )
