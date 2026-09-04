"""Level 2: RL 环境 — 加奖励函数和观察空间，可接入 rsl_rl 训练"""
from dataclasses import MISSING, dataclass
import torch
import warp as wp
import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab_arena.tasks.observations import observations as arena_obs
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass
from isaaclab_arena.assets.register import register_environment, register_task
from isaaclab_arena.environments.arena_environment_factory import (
    ArenaEnvironmentCfg, ArenaEnvironmentFactory,
)
from isaaclab_arena.environments.isaaclab_arena_environment import (
    IsaacLabArenaEnvironment,
)
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.utils.pose import Pose
import isaaclab_arena_examples.policy.base_rsl_rl_policy as base_rsl_rl_policy


# ═══ 终止条件函数 ═══
def cube_reached_target(env, object_cfg, target_pos, tolerance):
    """方块离目标位置 < tolerance 就成功"""
    object_pos = env.scene[object_cfg.name].data.root_pos_w
    # root_pos_w 是 warp proxy array，先取底层 torch tensor，再对齐设备
    object_pos_t = object_pos.torch if hasattr(object_pos, "torch") else object_pos
    target_pos_t = torch.as_tensor(target_pos, device=object_pos_t.device, dtype=object_pos_t.dtype)
    dist = torch.norm(object_pos_t - target_pos_t, dim=-1)
    return dist < tolerance


# ═══ 奖励函数 ═══
def distance_reward(env, object_cfg, ee_frame_cfg, target_pos):
    """末端离方块越近奖励越高（0→1）"""
    ee_pos = env.scene[ee_frame_cfg.name].data.target_pos_w[..., 0, :]
    object_pos = env.scene[object_cfg.name].data.root_pos_w
    ee_t = ee_pos.torch if hasattr(ee_pos, "torch") else ee_pos
    obj_t = object_pos.torch if hasattr(object_pos, "torch") else object_pos
    dist = torch.norm(ee_t - obj_t, dim=-1)
    return 1.0 / (1.0 + dist)


def ee_position_in_world_frame(env, ee_frame_cfg):
    """Return the end-effector frame position in world coordinates."""
    ee_frame = env.scene[ee_frame_cfg.name]
    return wp.to_torch(ee_frame.data.target_pos_w)[..., 0, :]


def success_bonus(env, object_cfg, target_pos, tolerance):
    """方块到达目标位置给 +10"""
    object_pos = env.scene[object_cfg.name].data.root_pos_w
    object_pos_t = object_pos.torch if hasattr(object_pos, "torch") else object_pos
    target_pos_t = torch.as_tensor(target_pos, device=object_pos_t.device, dtype=object_pos_t.dtype)
    dist = torch.norm(object_pos_t - target_pos_t, dim=-1)
    return (dist < tolerance).float() * 10.0


# ═══ 奖励 configclass ═══
@configclass
class MyRewardsCfg:
    reaching: RewardTermCfg = MISSING
    success: RewardTermCfg = MISSING

    def __init__(self, object_name: str, ee_frame_name: str, target_pos: tuple, tolerance: float):
        self.reaching = RewardTermCfg(
            func=distance_reward,
            params={
                "object_cfg": SceneEntityCfg(object_name),
                "ee_frame_cfg": SceneEntityCfg(ee_frame_name),
                "target_pos": target_pos,
            },
            weight=1.0,
        )
        self.success = RewardTermCfg(
            func=success_bonus,
            params={
                "object_cfg": SceneEntityCfg(object_name),
                "target_pos": target_pos,
                "tolerance": tolerance,
            },
            weight=1.0,
        )


# ═══ 观察 configclass ═══
@configclass
class MyObservationsCfg:
    # 用 task_obs 而不是 policy，避免与 FrankaObservationsCfg.policy 字段类型冲突
    task_obs: ObsGroup = MISSING

    def __init__(self, object_name: str, ee_frame_name: str):
        @configclass
        class TaskObsCfg(ObsGroup):
            # 物体在世界坐标系中的位置（使用 arena 提供的 warp-safe 函数）
            object_pos = ObsTerm(
                func=arena_obs.object_position_in_world_frame,
                params={"asset_cfg": SceneEntityCfg(object_name)},
            )
            # 末端执行器在世界坐标系中的位置（ee_frame 是 FrameTransformer，取 target_pos_w）
            eef_pos = ObsTerm(
                func=ee_position_in_world_frame,
                params={"ee_frame_cfg": SceneEntityCfg(ee_frame_name)},
            )

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True

        self.task_obs = TaskObsCfg()


# ═══ 终止条件 configclass ═══
@configclass
class MyTerminationsCfg:
    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out, time_out=True)
    success: TerminationTermCfg = MISSING

    def __init__(self, object_name: str, target_pos: tuple, tolerance: float):
        self.time_out = TerminationTermCfg(func=mdp_isaac_lab.time_out, time_out=True)
        self.success = TerminationTermCfg(
            func=cube_reached_target,
            params={
                "object_cfg": SceneEntityCfg(object_name),
                "target_pos": target_pos,
                "tolerance": tolerance,
            },
            time_out=False,
        )


# ═══ RL Task ═══
@register_task
class MyReachRLTask(TaskBase):
    def __init__(
        self,
        object_name: str,
        target_pos: tuple,
        ee_frame_name: str = "ee_frame",
        tolerance: float = 0.1,
    ):
        super().__init__(episode_length_s=10.0)

        self.termination_cfg = MyTerminationsCfg(object_name, target_pos, tolerance)
        self.rewards_cfg = MyRewardsCfg(object_name, ee_frame_name, target_pos, tolerance)
        self.observation_cfg = MyObservationsCfg(object_name, ee_frame_name)

    def get_scene_cfg(self):
        return InteractiveSceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=False)

    def get_termination_cfg(self):
        return self.termination_cfg

    def get_rewards_cfg(self):
        return self.rewards_cfg

    def get_observation_cfg(self):
        return self.observation_cfg

    def get_events_cfg(self):
        return None

    def get_metrics(self):
        return [SuccessRateMetric()]

    def get_mimic_env_cfg(self, arm_mode=None):
        return None


# ═══ 环境类 ═══
@dataclass
class Level2Cfg(ArenaEnvironmentCfg):
    robot: str = "franka_joint_pos"   # ★ RL 必须用关节位置控制，不能用 IK
    background: str = "table"
    object: str = "dex_cube"
    target_x: float = 0.3
    target_y: float = 0.0
    target_z: float = 0.3


@register_environment
class Level2RLReach(ArenaEnvironmentFactory[Level2Cfg]):
    name = "level2_rl_reach"
    _legacy_argparse_cfg_type = Level2Cfg

    def build(self, cfg: Level2Cfg) -> IsaacLabArenaEnvironment:
        background = self.asset_registry.get_asset_by_name(cfg.background)()
        robot = self.asset_registry.get_asset_by_name(cfg.robot)(
            concatenate_observation_terms=True
        )
        obj = self.asset_registry.get_asset_by_name(cfg.object)()
        light = self.asset_registry.get_asset_by_name("light")()

        obj.set_initial_pose(Pose(position_xyz=(0.1, 0.0, 0.05)))
        robot.set_initial_pose(Pose(position_xyz=(-0.4, 0.0, 0.0)))

        scene = Scene(assets=[background, light, obj])

        task = MyReachRLTask(
            object_name=cfg.object,
            target_pos=(cfg.target_x, cfg.target_y, cfg.target_z),
        )

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=robot,
            scene=scene,
            task=task,
            rl_framework_entry_point="rsl_rl_cfg_entry_point",
            rl_policy_cfg=f"{base_rsl_rl_policy.__name__}:RLPolicyCfg",
        )
