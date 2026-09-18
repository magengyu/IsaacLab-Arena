"""Level 0c: 当前版本 — 厨房 + 机器人 + 方块，需等 AWS 下载"""
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
class Level0cCfg(ArenaEnvironmentCfg):
    background: str = "kitchen"
    robot: str = "franka_ik"
    object: str = "dex_cube"


class Level0cKitchen(ArenaEnvironmentFactory[Level0cCfg]):
    name = "level0c_kitchen"
    _legacy_argparse_cfg_type = Level0cCfg

    def build(self, cfg: Level0cCfg) -> IsaacLabArenaEnvironment:
        background = self.asset_registry.get_asset_by_name(cfg.background)()
        robot = self.asset_registry.get_asset_by_name(cfg.robot)()
        obj = self.asset_registry.get_asset_by_name(cfg.object)()
        light = self.asset_registry.get_asset_by_name("light")()
        ground = self.asset_registry.get_asset_by_name("ground_plane")()

        obj.set_initial_pose(Pose(position_xyz=(0.4, 0.0, 0.05)))
        scene = Scene(assets=[ground, background, light, obj])

        return IsaacLabArenaEnvironment(
            name=self.name, embodiment=robot, scene=scene,
        )
