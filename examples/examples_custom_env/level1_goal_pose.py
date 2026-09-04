"""Level 1: 加入任务 — 方块到达目标位置就成功"""
from dataclasses import MISSING, dataclass
import torch
from isaaclab.envs import mdp as mdp_isaac_lab
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass
from isaaclab_arena.assets.register import register_environment, register_task
from isaaclab_arena.environments.arena_environment_factory import (
    ArenaEnvironmentCfg, ArenaEnvironmentFactory,
)
from isaaclab_arena.environments.isaaclab_arena_environment import (
    IsaacLabArenaEnvironment,
)
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.utils.pose import Pose


# ═══ 终止条件函数 ═══
def cube_reached_target(env, object_cfg, target_pos, tolerance):
    object_pos = env.scene[object_cfg.name].data.root_pos_w
    # root_pos_w 可能是 warp proxy array，需先取出底层 torch tensor
    object_pos_t = object_pos.torch if hasattr(object_pos, "torch") else object_pos
    # target_pos 是构造时传入的 CPU tensor，用 as_tensor 迁移到相同设备和 dtype
    target_pos_t = torch.as_tensor(target_pos, device=object_pos_t.device, dtype=object_pos_t.dtype)
    dist = torch.norm(object_pos_t - target_pos_t, dim=-1)
    return dist < tolerance


# ═══ Task ═══
@configclass
class MyTerminationsCfg:
    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)
    success: TerminationTermCfg = MISSING


@register_task
class MyGoalPoseTask(TaskBase):
    def __init__(self, object_name, target_pos, tolerance=0.1):
        super().__init__(episode_length_s=10.0)

        self.termination_cfg = MyTerminationsCfg(
            success=TerminationTermCfg(
                func=cube_reached_target,
                params={
                    "object_cfg": SceneEntityCfg(object_name),
                    "target_pos": target_pos,
                    "tolerance": tolerance,
                },
                time_out=False,
            ),
        )

    def get_scene_cfg(self):       return InteractiveSceneCfg(num_envs=1, env_spacing=3.0)
    def get_termination_cfg(self): return self.termination_cfg
    def get_events_cfg(self):      return None
    def get_metrics(self):         return []
    def get_mimic_env_cfg(self, arm_mode=None): return None


# ═══ 环境类 ═══
@dataclass
class Level1Cfg(ArenaEnvironmentCfg):
    robot: str = "franka_ik"
    background: str = "table"
    object: str = "dex_cube"
    target_x: float = 0.3
    target_y: float = 0.0
    target_z: float = 0.3


@register_environment
class Level1GoalPose(ArenaEnvironmentFactory[Level1Cfg]):
    name = "level1_goal_pose"
    _legacy_argparse_cfg_type = Level1Cfg

    def build(self, cfg: Level1Cfg) -> IsaacLabArenaEnvironment:
        background = self.asset_registry.get_asset_by_name(cfg.background)()
        robot = self.asset_registry.get_asset_by_name(cfg.robot)()
        obj = self.asset_registry.get_asset_by_name(cfg.object)()
        light = self.asset_registry.get_asset_by_name("light")()

        obj.set_initial_pose(Pose(position_xyz=(0.1, 0.0, 0.05)))
        robot.set_initial_pose(Pose(position_xyz=(-0.4, 0.0, 0.0)))
        scene = Scene(assets=[background, light, obj])

        task = MyGoalPoseTask(
            object_name=cfg.object,
            target_pos=(cfg.target_x, cfg.target_y, cfg.target_z),
        )

        return IsaacLabArenaEnvironment(
            name=self.name, embodiment=robot, scene=scene, task=task,
        )
