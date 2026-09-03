# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import ActionTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.manipulation.stack.mdp.observations import ee_frame_pos, ee_frame_quat

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.utils.pose import Pose

_SIMPLE_ARM_JOINT_NAMES = (
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
)

_SIMPLE_ARM_READY_POSE = {
    "joint_1": 0.0,
    "joint_2": 0.35,
    "joint_3": -0.45,
    "joint_4": 0.0,
    "joint_5": 0.35,
    "joint_6": 0.0,
}


def _find_simple_arm_usd_path(require_exists: bool = False) -> str:
    relative_paths = (
        Path("examples/examples_urdf_all_train/assets/simple_urdf_arm/usd/simple_6dof_arm/simple_6dof_arm.usda"),
        Path("examples/examples_urdf_all_train/assets/simple_urdf_arm/usd/simple_6dof_arm_1/simple_6dof_arm.usda"),
    )
    for parent in Path(__file__).resolve().parents:
        for relative_path in relative_paths:
            candidate = parent / relative_path
            if candidate.exists():
                return str(candidate)
    fallback_path = Path(__file__).resolve().parents[2] / relative_paths[0]
    if require_exists:
        search_locations = ", ".join(str(path) for path in relative_paths)
        raise FileNotFoundError(f"Simple URDF arm USD not found. Searched: {search_locations}")
    return str(fallback_path)


@register_asset
class SimpleURDFArmIKEmbodiment(EmbodimentBase):
    """A tiny primitive-shape 6-DOF arm for URDF/USD/teleop teaching."""

    name = "simple_urdf_arm_ik"
    tags = ["embodiment"]
    default_arm_mode = ArmMode.SINGLE_ARM

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        initial_joint_pose: list[float] | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
    ):
        super().__init__(enable_cameras, initial_pose, concatenate_observation_terms, arm_mode)
        self.scene_config = SimpleURDFArmSceneCfg()
        usd_path = _find_simple_arm_usd_path(require_exists=True)
        self.scene_config.robot.spawn = self.scene_config.robot.spawn.replace(usd_path=usd_path)
        print(f"[INFO] Simple URDF arm USD: {usd_path}")
        self.observation_config = SimpleURDFArmObservationsCfg()
        self.observation_config.policy.concatenate_terms = self.concatenate_observation_terms
        self.reward_config = SimpleURDFArmRewardsCfg()
        self.action_config = SimpleURDFArmIKActionCfg()

        if initial_joint_pose is not None:
            self.set_initial_joint_pose(initial_joint_pose)

    def set_initial_joint_pose(self, initial_joint_pose: list[float]) -> None:
        expected_joint_count = len(_SIMPLE_ARM_JOINT_NAMES)
        assert (
            len(initial_joint_pose) == expected_joint_count
        ), f"expected {expected_joint_count} joint positions, got {len(initial_joint_pose)}"
        self.scene_config.robot.init_state = self.scene_config.robot.init_state.replace(
            joint_pos=dict(zip(_SIMPLE_ARM_JOINT_NAMES, initial_joint_pose))
        )

    def get_command_body_name(self) -> str:
        return self.action_config.arm_action.body_name

    def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
        return "ee_frame"

    def get_teleop_target_frame_prim_path(self) -> str:
        return "/World/envs/env_0/Robot/Geometry/base_link"


@configclass
class SimpleURDFArmSceneCfg:
    """Scene additions for the tiny URDF arm."""

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_find_simple_arm_usd_path(),
            activate_contact_sensors=False,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos=_SIMPLE_ARM_READY_POSE,
        ),
        soft_joint_pos_limit_factor=1.0,
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint_[1-6]"],
                effort_limit=40.0,
                velocity_limit=2.5,
                stiffness=120.0,
                damping=12.0,
            ),
        },
    )

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link/link_1/link_2/link_3/link_4/link_5/link_6",
                name="end_effector",
            ),
        ],
    )

    def __post_init__(self):
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/SimpleURDFArmFrameTransformer"
        self.ee_frame.visualizer_cfg = marker_cfg


@configclass
class SimpleURDFArmIKActionCfg:
    """6D relative pose action for the tiny arm."""

    arm_action: ActionTermCfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["joint_[1-6]"],
        body_name="link_6",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.12,
    )


@configclass
class SimpleURDFArmObservationsCfg:
    """Observation specifications for the tiny URDF arm."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        eef_pos = ObsTerm(func=ee_frame_pos)
        eef_quat = ObsTerm(func=ee_frame_quat)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class SimpleURDFArmRewardsCfg:
    """Small regularization rewards so ManagerBased env construction has reward terms."""

    action_rate = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.0001)
    joint_vel = RewardTermCfg(func=mdp.joint_vel_l2, weight=-0.0001, params={"asset_cfg": SceneEntityCfg("robot")})
