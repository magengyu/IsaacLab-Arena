"""Level 0a: 最简 — 地面 + 机器人，无物体，秒开"""
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
class Level0aCfg(ArenaEnvironmentCfg):
    robot: str = "franka_ik"


class Level0aMinimal(ArenaEnvironmentFactory[Level0aCfg]):
    name = "level0a_minimal"
    _legacy_argparse_cfg_type = Level0aCfg

    def build(self, cfg: Level0aCfg) -> IsaacLabArenaEnvironment:
        robot = self.asset_registry.get_asset_by_name(cfg.robot)()
        ground = self.asset_registry.get_asset_by_name("ground_plane")()
        light = self.asset_registry.get_asset_by_name("light")()
        scene = Scene(assets=[ground, light])
        return IsaacLabArenaEnvironment(
            name=self.name, embodiment=robot, scene=scene,
        )
