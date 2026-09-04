# Level 0d（全可配置）：背景 / 机器人 / 物体都能通过 CLI 覆盖
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
class Level0Cfg(ArenaEnvironmentCfg):
    """配置：改背景/机器人/物体靠改这些字段"""
    background: str = "kitchen"       # 背景场景名
    robot: str = "franka_ik"         # 机器人名
    object: str = "dex_cube"         # 物体名


class Level0HelloWorld(ArenaEnvironmentFactory[Level0Cfg]):
    name = "level0_hello_world"
    _legacy_argparse_cfg_type = Level0Cfg

    def build(self, cfg: Level0Cfg) -> IsaacLabArenaEnvironment:
        # 1. 从注册表取资产
        background = self.asset_registry.get_asset_by_name(cfg.background)()
        robot = self.asset_registry.get_asset_by_name(cfg.robot)()
        cube = self.asset_registry.get_asset_by_name(cfg.object)()
        light = self.asset_registry.get_asset_by_name("light")()

        # 2. 设置物体位置（机械臂前面）
        cube.set_initial_pose(Pose(position_xyz=(0.4, 0.0, 0.05)))

        # 3. 组装场景
        scene = Scene(assets=[background, light, cube])

        # 4. 返回（无 Task）
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=robot,
            scene=scene,
        )
