"""Level 4: 复合任务链 — 先抓后放

演示 SequentialTaskBase 的用法：
  子任务 1（LiftObjectTask）  →  机械臂把方块举起来（高于初始位置 25 cm）
  子任务 2（GoalPoseTask）    →  把方块移到指定目标区域（"放置区"）

关键收益：
  - SequentialTaskBase 自动管理子任务状态机（current_subtask_idx）
  - 合并各子任务的终止条件/事件/指标，生成统一的 ManagerBased 环境 cfg
  - 全程使用内置 Isaac Sim USD（无 Nucleus 远程下载），启动快

注意：GoalPoseTask 不使用接触传感器，仅判断方块质心是否进入目标长方体区域，
与 PickAndPlaceTask（需要接触检测）相比更简洁，非常适合学习阶段使用。
"""
from dataclasses import dataclass

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import (
    ArenaEnvironmentCfg,
    ArenaEnvironmentFactory,
)
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.tasks.goal_pose_task import GoalPoseTask
from isaaclab_arena.tasks.lift_object_task import LiftObjectTask
from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase
from isaaclab_arena.utils.pose import Pose


# ═══ 环境配置 ═══
@dataclass
class Level4Cfg(ArenaEnvironmentCfg):
    robot: str = "franka_ik"
    pick_object: str = "dex_cube"


# ═══ 环境类 ═══
@register_environment
class Level4PickAndPlace(ArenaEnvironmentFactory[Level4Cfg]):
    """Level 4：顺序复合任务 — 先把方块举起，再移到目标放置区。

    状态机：
        [0] LiftObjectTask   ← 举起方块 25 cm  →  成功后切换到
        [1] GoalPoseTask     ← 把方块送入目标区域（x: 0.25~0.55, y: 0.15~0.45, z: 0.1~0.5）
    """

    name = "level4_pick_and_place"
    _legacy_argparse_cfg_type = Level4Cfg

    def build(self, cfg: Level4Cfg) -> IsaacLabArenaEnvironment:
        # ─── Step 1: 实例化资产 ───
        table    = self.asset_registry.get_asset_by_name("table")()
        robot    = self.asset_registry.get_asset_by_name(cfg.robot)()
        dex_cube = self.asset_registry.get_asset_by_name(cfg.pick_object)()
        light    = self.asset_registry.get_asset_by_name("light")()

        # ─── Step 2: 设置初始位姿 ───
        table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0)))
        robot.set_initial_pose(Pose(position_xyz=(-0.4, 0.0, 0.0)))
        # 方块放在桌面中心偏右（z ≈ 0.04 是方块半边长）
        dex_cube.set_initial_pose(Pose(position_xyz=(0.4, 0.0, 0.04)))

        # ─── Step 3: 定义两个子任务 ───

        # 子任务 1：把方块举起 25 cm（相对于初始位姿的 delta）
        subtask_lift = LiftObjectTask(
            lift_object=dex_cube,
            background_scene=table,
            episode_length_s=10.0,
            goal_position_delta_xyz=(0.0, 0.0, 0.25),
            goal_position_tolerance=0.06,
        )

        # 子任务 2：把方块移到目标放置区（x: 0.25~0.55 m, y: 0.15~0.45 m, z: 0.1~0.5 m）
        # 这个区域相当于把方块从正前方移到右前方桌面某处
        subtask_place = GoalPoseTask(
            object=dex_cube,
            episode_length_s=10.0,
            target_x_range=(0.25, 0.55),
            target_y_range=(0.15, 0.45),
            target_z_range=(0.1, 0.5),
        )

        # ─── Step 4: 用 SequentialTaskBase 组合成顺序复合任务 ───
        # desired_subtask_success_state=[True, True] → 两个子任务都必须先后成功
        task = SequentialTaskBase(
            subtasks=[subtask_lift, subtask_place],
            episode_length_s=20.0,
            desired_subtask_success_state=[True, True],
        )

        # ─── Step 5: 组装场景 ───
        scene = Scene(assets=[table, light, dex_cube])

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=robot,
            scene=scene,
            task=task,
        )
