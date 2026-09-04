"""Level 5: 顺序复合 RL 环境 — 先举起再放置，可接入 rsl_rl 训练

在 Level 4 的顺序任务基础上，加入完整的 RL 奖励函数和观察空间：

  子任务 1（LiftObjectTaskRL）→ 举起方块，驱动奖励：
      reaching_object           +1   末端靠近物体
      lifting_object            +15  物体高于 minimum_height
      object_goal_tracking      +16  物体靠近随机采样目标点（命令目标）
      object_goal_tracking_fine +5   精细对齐奖励

  子任务 2（GoalPoseTask）      → 成功判定：方块进入目标区域

关键设计：
  - SequentialTaskBase 管理两阶段状态机（current_subtask_idx）和组合终止条件
  - Level5SequentialRLTask 覆写 get_rewards_cfg / get_observation_cfg / get_commands_cfg，
    直接复用 LiftObjectTaskRL 的 RL 配置——无需重写奖励函数
  - 统一命令目标（UniformPoseCommand）随机采样放置位，两阶段共享同一 obs/reward
"""
from dataclasses import dataclass, field

import isaaclab_arena_examples.policy.base_rsl_rl_policy as base_rsl_rl_policy

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import (
    ArenaEnvironmentCfg,
    ArenaEnvironmentFactory,
)
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.tasks.goal_pose_task import GoalPoseTask
from isaaclab_arena.tasks.lift_object_task import LiftObjectTaskRL
from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.utils.pose import Pose


# ═══ 复合 RL Task ═══
class Level5SequentialRLTask(SequentialTaskBase):
    """顺序复合 RL 任务：继承 SequentialTaskBase 的状态机逻辑，
    覆写 RL 三要素（奖励 / 观察 / 命令）。

    复用 LiftObjectTaskRL 已完整配置的 RL 栈：
      - 奖励：reaching_object / lifting_object / object_goal_tracking
      - 观察：目标点位置 + 物体相对机器人坐标
      - 命令：UniformPoseCommand（每 5 秒随机采样新目标点）
    """

    def __init__(
        self,
        subtasks: list[TaskBase],
        subtask_lift: LiftObjectTaskRL,
        episode_length_s: float = 20.0,
    ):
        super().__init__(
            subtasks=subtasks,
            episode_length_s=episode_length_s,
            desired_subtask_success_state=[True, True],
        )
        # 保存对 LiftObjectTaskRL 的引用，用于委托 RL 配置
        self._rl_subtask = subtask_lift

    # ★ 覆写：让父类的 get_rewards_cfg() 不再返回 None
    def get_rewards_cfg(self):
        return self._rl_subtask.get_rewards_cfg()

    # ★ 覆写：让父类的 get_observation_cfg() 不再返回 None
    def get_observation_cfg(self):
        return self._rl_subtask.get_observation_cfg()

    # ★ 覆写：命令管理器（用于随机采样目标位置）
    def get_commands_cfg(self):
        return self._rl_subtask.get_commands_cfg()


# ═══ Level 5 专用 RL 配置 ═══
@configclass
class Level5PolicyCfg(base_rsl_rl_policy.RLPolicyCfg):
    """基于通用 ``RLPolicyCfg``，只调两个旋钮防止训练发散：

    - 开启 actor/critic 观测归一化（原默认 False）
    - 降低 value loss 系数 1.0 → 0.5（配合降奖励，避免 critic 输出爆炸）
    """

    policy: RslRlPpoActorCriticCfg = field(default_factory=lambda: RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    ))
    algorithm: RslRlPpoAlgorithmCfg = field(default_factory=lambda: RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=0.0001,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    ))


# ═══ 环境配置 ═══
@dataclass
class Level5Cfg(ArenaEnvironmentCfg):
    robot: str = "franka_joint_pos"   # ★ RL 必须用关节位置控制
    pick_object: str = "dex_cube"


# ═══ 环境类 ═══
@register_environment
class Level5SequentialRL(ArenaEnvironmentFactory[Level5Cfg]):
    """Level 5：可训练的顺序复合 RL 环境。

    阶段 1：举起方块（LiftObjectTaskRL 成功判定：物体高于 minimum_height）
    阶段 2：移到目标区域（GoalPoseTask 成功判定：物体进入目标长方体）

    训练命令（在容器内执行）：
        /isaac-sim/python.sh submodules/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \\
            --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \\
            --task level5_sequential_rl \\
            --num_envs 256 --max_iterations 1000
    """

    name = "level5_sequential_rl"
    _legacy_argparse_cfg_type = Level5Cfg

    def build(self, cfg: Level5Cfg) -> IsaacLabArenaEnvironment:
        # ─── Step 1: 实例化资产 ───
        table    = self.asset_registry.get_asset_by_name("table")()
        robot    = self.asset_registry.get_asset_by_name(cfg.robot)(
            concatenate_observation_terms=True  # RL 必须
        )
        dex_cube = self.asset_registry.get_asset_by_name(cfg.pick_object)()
        light    = self.asset_registry.get_asset_by_name("light")()

        # ─── Step 2: 设置初始位姿 ───
        table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0)))
        robot.set_initial_pose(Pose(position_xyz=(-0.4, 0.0, 0.0)))
        # 方块放在桌面（z ≈ 0.04 是方块半边长）
        dex_cube.set_initial_pose(Pose(position_xyz=(0.4, 0.0, 0.04)))

        # ─── Step 3: 定义两个子任务 ───

        # 子任务 1：RL 举起任务（自带奖励 / 观察 / 命令配置）
        #   target_z_delta=(0.2, 0.4) → 随机采样目标点在初始位置上方 20~40 cm
        subtask_lift = LiftObjectTaskRL(
            lift_object=dex_cube,
            background_scene=table,
            embodiment=robot,
            minimum_height_to_lift=0.04,
            episode_length_s=10.0,
            rl_training_mode=True,
            target_x_delta=(-0.1, 0.1),
            target_y_delta=(-0.15, 0.15),
            target_z_delta=(0.2, 0.4),
        )

        # ★ 降奖励量级：防单步 return 过大 → GAE 返回爆炸 → critic 发散 → NaN
        subtask_lift.rewards_cfg.lifting_object.weight = 2.0
        subtask_lift.rewards_cfg.object_goal_tracking.weight = 2.0
        subtask_lift.rewards_cfg.object_goal_tracking_fine_grained.weight = 1.0

        # 子任务 2：目标区域判定（举起后把方块移到指定区域即成功）
        subtask_place = GoalPoseTask(
            object=dex_cube,
            episode_length_s=10.0,
            target_x_range=(0.25, 0.55),
            target_y_range=(0.15, 0.45),
            target_z_range=(0.1, 0.5),
        )

        # ─── Step 4: 组合成可训练的顺序复合任务 ───
        task = Level5SequentialRLTask(
            subtasks=[subtask_lift, subtask_place],
            subtask_lift=subtask_lift,
            episode_length_s=20.0,
        )

        # ─── Step 5: 组装场景 ───
        scene = Scene(assets=[table, light, dex_cube])

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=robot,
            scene=scene,
            task=task,
            # ★ RL 框架绑定
            rl_framework_entry_point="rsl_rl_cfg_entry_point",
            rl_policy_cfg=f"{__name__}:Level5PolicyCfg",
        )
