"""Level 3: 自定义物体 + 关系放置

新注册一个 MyCustomBlock 资产（复用 DexCube 的 USD），
然后用 On / NextTo 关系让 Arena 自动计算摆放位置，无需手写坐标。
"""
from dataclasses import dataclass

from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.assets.register import register_asset, register_environment
from isaaclab_arena.environments.arena_environment_factory import (
    ArenaEnvironmentCfg,
    ArenaEnvironmentFactory,
)
from isaaclab_arena.environments.isaaclab_arena_environment import (
    IsaacLabArenaEnvironment,
)
from isaaclab_arena.relations.relations import IsAnchor, NextTo, On
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.utils.pose import Pose


# ═══ Step 1: 注册自定义物体 ═══
@register_asset
class MyCustomBlock(LibraryObject):
    """自定义方块：复用 Isaac Sim 内置 DexCube USD，演示 @register_asset 用法。"""

    name = "my_custom_block"
    tags = ["object", "my_objects"]
    # 直接引用 Isaac Sim Nucleus 里已有的 USD 文件
    usd_path = (
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
        "Assets/Isaac/6.0/Isaac/Props/Blocks/DexCube/dex_cube_instanceable.usd"
    )


# ═══ Step 2: 环境类 ═══
@dataclass
class Level3Cfg(ArenaEnvironmentCfg):
    robot: str = "franka_ik"


@register_environment
class Level3CustomObject(ArenaEnvironmentFactory[Level3Cfg]):
    name = "level3_custom_object"
    _legacy_argparse_cfg_type = Level3Cfg

    def build(self, cfg: Level3Cfg) -> IsaacLabArenaEnvironment:
        # 资产实例化
        table = self.asset_registry.get_asset_by_name("table")()
        robot = self.asset_registry.get_asset_by_name(cfg.robot)()
        cube1 = self.asset_registry.get_asset_by_name("my_custom_block")()
        cube2 = self.asset_registry.get_asset_by_name("dex_cube")()
        light = self.asset_registry.get_asset_by_name("light")()

        # 桌子固定在原点，机器人放在桌子左侧
        table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0)))
        robot.set_initial_pose(Pose(position_xyz=(-0.4, 0.0, 0.0)))

        # ★ 关系：table 作为锚点（固定参考），cube1 放在桌面上，cube2 紧靠 cube1 右侧
        table.add_relation(IsAnchor())
        cube1.add_relation(On(table, clearance_m=0.01))
        cube2.add_relation(NextTo(cube1, distance_m=0.15))

        scene = Scene(assets=[table, light, cube1, cube2])

        return IsaacLabArenaEnvironment(
            name=self.name, embodiment=robot, scene=scene,
        )
