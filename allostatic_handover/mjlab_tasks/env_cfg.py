"""Mjlab environment configs for allostatic handover."""

from __future__ import annotations

import copy

import mujoco

from allostatic_handover.envs.human_hidden_state import HumanState
from allostatic_handover.mjlab_tasks.hrgym_assets import (
  hrgym_hammer_spec,
  hrgym_human_spec,
  hrgym_table_spec,
)
from allostatic_handover.world_model.belief_model import DEFAULT_BELIEF_MODEL_PATH
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import DifferentialIKActionCfg, JointPositionActionCfg
from mjlab.asset_zoo.robots.i2rt_yam.yam_constants import (
  FULL_COLLISION as YAM_FULL_COLLISION,
)
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers import MetricsTermCfg, ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.manipulation.config.yam.env_cfgs import yam_lift_cube_env_cfg
from mjlab.tasks.velocity import mdp as velocity_mdp

from . import mdp

FULL_TABLE_TOP_Z = 0.845
HRGYM_REFERENCE_ROBOT_HAND_POS = (0.31649374, -0.01341937, 1.31636695)
FULL_YAM_ROOT_QUAT = (1.0, 0.0, 0.0, 0.0)
FULL_YAM_GRASPED_START_ROOT_QUAT = (0.70710678, 0.0, 0.0, 0.70710678)
# The Yam root frame is the tabletop placement reference in the upstream
# Mjlab-Lift-Cube-Yam task. Keep it on the HRGym table surface so the base
# plate is visible on the table rather than hidden in the tabletop.
FULL_YAM_TABLE_SURFACE_ROOT_Z = FULL_TABLE_TOP_Z
FULL_YAM_ROOT_POS = (0.20, 0.0, FULL_YAM_TABLE_SURFACE_ROOT_Z)
FULL_YAM_GRASPED_START_ROOT_POS = (0.43649374, 0.09124467, 1.31460755)
FULL_YAM_PEDESTAL_HALF_SIZE = (
  0.18,
  0.18,
  0.005,
)
FULL_YAM_PEDESTAL_POS = (
  FULL_YAM_ROOT_POS[0],
  FULL_YAM_ROOT_POS[1],
  FULL_TABLE_TOP_Z - FULL_YAM_PEDESTAL_HALF_SIZE[2],
)
FULL_YAM_GRASPED_START_PEDESTAL_HALF_SIZE = (
  0.18,
  0.18,
  (FULL_YAM_GRASPED_START_ROOT_POS[2] - FULL_TABLE_TOP_Z) / 2.0,
)
FULL_YAM_GRASPED_START_PEDESTAL_POS = (
  FULL_YAM_GRASPED_START_ROOT_POS[0],
  FULL_YAM_GRASPED_START_ROOT_POS[1],
  FULL_TABLE_TOP_Z + FULL_YAM_GRASPED_START_PEDESTAL_HALF_SIZE[2],
)
FULL_HRGYM_OBJECT_Z = 0.97084304
FULL_YAM_OBJECT_Z = FULL_TABLE_TOP_Z + 0.035
# HRGym's reach-out animation already brings the hand into Yam's workspace.
# Keep the human at the HRGym world pose and only adapt the Yam/object layout.
FULL_YAM_HUMAN_BASE_POS_OFFSET = (0.0, 0.0, 0.0)
# Fixed tabletop hammer pose for the baseline/full table-start task.  The pose
# stays inside Yam's reachable workspace but away from the human reach target,
# so success requires an actual pick/transport/release rather than a near-start
# shortcut.
FULL_YAM_OBJECT_FIXED_XY = (0.50, -0.16)
FULL_YAM_OBJECT_X_RANGE = (FULL_YAM_OBJECT_FIXED_XY[0], FULL_YAM_OBJECT_FIXED_XY[0])
FULL_YAM_OBJECT_Y_RANGE = (FULL_YAM_OBJECT_FIXED_XY[1], FULL_YAM_OBJECT_FIXED_XY[1])
FULL_YAM_OBJECT_YAW_RANGE = (1.57079632679, 1.57079632679)
FULL_YAM_OBJECT_INIT_POS = (
  FULL_YAM_OBJECT_FIXED_XY[0],
  FULL_YAM_OBJECT_FIXED_XY[1],
  FULL_YAM_OBJECT_Z,
)
# Matches the command reset rotation generated from euler xyz=(0, pi, pi/2).
# Keeping the entity init pose aligned with reset avoids a one-frame visual snap
# from the hammer asset's identity orientation to the fixed tabletop yaw.
FULL_YAM_OBJECT_INIT_ROT = (0.0, -0.70710678, 0.70710678, 0.0)
FULL_YAM_MIN_GRASP_SITE_Z = FULL_TABLE_TOP_Z + 0.05
# The hammer root is near the gripper grasp point.  Handover success should be
# checked at the offered side of the hammer, not at the grasp/root point.
FULL_YAM_HANDOFF_OBJECT_OFFSET = (0.18, 0.06, 0.08)


def _set_yam_root_pose(
  robot: EntityCfg,
  pos: tuple[float, float, float],
  rot: tuple[float, float, float, float] = FULL_YAM_ROOT_QUAT,
) -> None:
  # Mjlab's Yam config reuses the module-level HOME_KEYFRAME object as the
  # initial state. Mutating it directly contaminates already-registered task
  # configs, so each allostatic task must own a private copy before overrides.
  robot.init_state = copy.deepcopy(robot.init_state)
  robot.init_state.pos = pos
  robot.init_state.rot = rot


def _primitive_entity_spec(
  name: str,
  geom_type: mujoco.mjtGeom,
  size: tuple[float, ...],
  rgba: tuple[float, float, float, float],
  mass: float = 0.01,
) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name=name)
  body.gravcomp = 1.0
  body.add_freejoint(name=f"{name}_joint")
  geom = body.add_geom(
    name=f"{name}_geom",
    type=geom_type,
    size=size,
    mass=mass,
    rgba=rgba,
  )
  geom.contype = 0
  geom.conaffinity = 0
  return spec


def _human_torso_spec() -> mujoco.MjSpec:
  return _primitive_entity_spec(
    "human_torso",
    mujoco.mjtGeom.mjGEOM_BOX,
    (0.09, 0.06, 0.16),
    (0.38, 0.45, 0.55, 0.95),
  )


def _human_upper_arm_spec() -> mujoco.MjSpec:
  return _primitive_entity_spec(
    "human_upper_arm",
    mujoco.mjtGeom.mjGEOM_BOX,
    (0.035, 0.035, 0.12),
    (0.76, 0.60, 0.48, 0.95),
  )


def _human_forearm_spec() -> mujoco.MjSpec:
  return _primitive_entity_spec(
    "human_forearm",
    mujoco.mjtGeom.mjGEOM_BOX,
    (0.03, 0.03, 0.13),
    (0.82, 0.66, 0.54, 0.95),
  )


def _human_hand_spec() -> mujoco.MjSpec:
  return _primitive_entity_spec(
    "human_hand",
    mujoco.mjtGeom.mjGEOM_SPHERE,
    (0.045,),
    (0.96, 0.74, 0.58, 0.95),
  )


def _yam_pedestal_spec(
  half_size: tuple[float, float, float] = FULL_YAM_PEDESTAL_HALF_SIZE,
) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="yam_pedestal")
  body.add_geom(
    name="yam_pedestal_collision",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=half_size,
    rgba=(0.32, 0.32, 0.34, 1.0),
    mass=0.0,
  )
  body.add_site(
    name="yam_pedestal_top",
    pos=(0.0, 0.0, half_size[2]),
    size=(0.01,),
    rgba=(0.0, 0.0, 0.0, 0.0),
  )
  return spec


def _yam_grasped_start_pedestal_spec() -> mujoco.MjSpec:
  return _yam_pedestal_spec(FULL_YAM_GRASPED_START_PEDESTAL_HALF_SIZE)


def allostatic_handover_yam_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Yam-based Mjlab allostatic handover task config."""
  cfg = yam_lift_cube_env_cfg(play=play)

  cfg.scene.entities = {
    **cfg.scene.entities,
    "human_torso": EntityCfg(spec_fn=_human_torso_spec),
    "human_upper_arm": EntityCfg(spec_fn=_human_upper_arm_spec),
    "human_forearm": EntityCfg(spec_fn=_human_forearm_spec),
    "human_hand": EntityCfg(spec_fn=_human_hand_spec),
  }
  cfg.scene.env_spacing = 1.4
  cfg.scale_rewards_by_dt = False
  cfg.curriculum = {}

  cfg.actions = {
    "arm_ik": DifferentialIKActionCfg(
      entity_name="robot",
      actuator_names=("joint[1-6]",),
      frame_type="site",
      frame_name="grasp_site",
      use_relative_mode=True,
      delta_pos_scale=0.045,
      orientation_weight=0.0,
      damping=0.055,
      max_dq=0.12,
      posture_weight=0.015,
    ),
    "gripper": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=("left_finger",),
      scale=0.01875,
      offset=0.01875,
      clip={"left_finger": (0.0, 0.0375)},
      use_default_offset=False,
    ),
    "speech": mdp.SpeechActionCfg(entity_name="robot"),
  }

  cfg.commands = {
    "handover": mdp.AllostaticHandoverCommandCfg(
      resampling_time_range=(1000.0, 1000.0),
      debug_vis=True,
    )
  }

  ee_cfg = SceneEntityCfg("robot", site_names=("grasp_site",))
  robot_joints_cfg = SceneEntityCfg("robot", joint_names=(".*",))
  actor_terms = {
    "joint_pos": ObservationTermCfg(
      func=velocity_mdp.joint_pos_rel,
      params={"asset_cfg": robot_joints_cfg},
    ),
    "joint_vel": ObservationTermCfg(
      func=velocity_mdp.joint_vel_rel,
      params={"asset_cfg": robot_joints_cfg},
    ),
    "ee_to_cube": ObservationTermCfg(
      func=manipulation_mdp.ee_to_object_distance,
      params={"object_name": "cube", "asset_cfg": ee_cfg},
    ),
    "ee_to_hand": ObservationTermCfg(
      func=mdp.ee_to_hand,
      params={"command_name": "handover", "asset_cfg": ee_cfg},
    ),
    "cube_to_hand": ObservationTermCfg(
      func=mdp.object_to_hand,
      params={"object_name": "cube", "command_name": "handover"},
    ),
    "readiness_belief": ObservationTermCfg(
      func=mdp.readiness_belief,
      params={"command_name": "handover"},
    ),
    "load_proxy": ObservationTermCfg(
      func=mdp.load_proxy,
      params={"command_name": "handover"},
    ),
    "speech_context": ObservationTermCfg(
      func=mdp.speech_context,
      params={"command_name": "handover"},
    ),
    "phase_progress": ObservationTermCfg(
      func=mdp.phase_progress,
      params={"command_name": "handover"},
    ),
    "actions": ObservationTermCfg(func=velocity_mdp.last_action),
  }
  critic_terms = {
    **actor_terms,
    "privileged_human_state": ObservationTermCfg(
      func=mdp.privileged_human_state,
      params={"command_name": "handover"},
    ),
    "privileged_load": ObservationTermCfg(
      func=mdp.privileged_load,
      params={"command_name": "handover"},
    ),
  }
  cfg.observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=False),
    "critic": ObservationGroupCfg(critic_terms, enable_corruption=False),
  }

  cfg.rewards = {
    "handover": RewardTermCfg(
      func=mdp.staged_handover_reward,
      weight=1.0,
      params={
        "command_name": "handover",
        "object_name": "cube",
        "reaching_std": 0.18,
        "handoff_std": 0.20,
        "asset_cfg": ee_cfg,
      },
    ),
    "handover_precise": RewardTermCfg(
      func=mdp.handover_precision_reward,
      weight=0.8,
      params={
        "command_name": "handover",
        "object_name": "cube",
        "std": 0.07,
      },
    ),
    "success": RewardTermCfg(
      func=mdp.success_bonus,
      weight=8.0,
      params={"command_name": "handover"},
    ),
    "speech_penalty": RewardTermCfg(
      func=mdp.speech_penalty,
      weight=-0.04,
      params={"command_name": "handover"},
    ),
    "allostatic_load": RewardTermCfg(
      func=mdp.allostatic_load_penalty,
      weight=-0.10,
      params={"command_name": "handover"},
    ),
    "waiting_cost": RewardTermCfg(
      func=mdp.waiting_cost,
      weight=-0.25,
      params={"command_name": "handover"},
    ),
    "action_rate_l2": RewardTermCfg(func=velocity_mdp.action_rate_l2, weight=-0.01),
    "joint_pos_limits": RewardTermCfg(
      func=velocity_mdp.joint_pos_limits,
      weight=-5.0,
      params={"asset_cfg": robot_joints_cfg},
    ),
  }

  cfg.terminations = {
    "time_out": TerminationTermCfg(func=velocity_mdp.time_out, time_out=True),
    "handover_complete": TerminationTermCfg(
      func=mdp.handover_complete,
      params={"command_name": "handover"},
    ),
  }

  metric_params = {"command_name": "handover"}
  cfg.metrics = {
    "success": MetricsTermCfg(
      func=mdp.success,
      params=metric_params,
      reduce="last",
    ),
    "speech/robot_speech_count": MetricsTermCfg(
      func=mdp.robot_speech_count,
      params=metric_params,
      reduce="last",
    ),
    "speech/silence_ratio": MetricsTermCfg(
      func=mdp.silence_ratio,
      params=metric_params,
      reduce="last",
    ),
    "speech/repeated_speech_count": MetricsTermCfg(
      func=mdp.repeated_speech_count,
      params=metric_params,
      reduce="last",
    ),
    "human_readiness": MetricsTermCfg(func=mdp.human_readiness, params=metric_params),
    "human_reach_progress": MetricsTermCfg(
      func=mdp.human_reach_progress,
      params=metric_params,
    ),
    "human_state/ready_ratio": MetricsTermCfg(
      func=mdp.human_state_ratio,
      params={**metric_params, "state": HumanState.READY},
    ),
    "human_state/hesitant_ratio": MetricsTermCfg(
      func=mdp.human_state_ratio,
      params={**metric_params, "state": HumanState.HESITANT},
    ),
    "human_state/overloaded_ratio": MetricsTermCfg(
      func=mdp.human_state_ratio,
      params={**metric_params, "state": HumanState.OVERLOADED},
    ),
    "human_state/withdrawing_ratio": MetricsTermCfg(
      func=mdp.human_state_ratio,
      params={**metric_params, "state": HumanState.WITHDRAWING},
    ),
    "human_state/grasping_ratio": MetricsTermCfg(
      func=mdp.human_state_ratio,
      params={**metric_params, "state": HumanState.GRASPING},
    ),
    "allostasis/load_mean": MetricsTermCfg(
      func=mdp.allostatic_load_total,
      params=metric_params,
    ),
    "allostasis/attention_load": MetricsTermCfg(
      func=mdp.attention_load,
      params=metric_params,
    ),
    "allostasis/turn_taking_load": MetricsTermCfg(
      func=mdp.turn_taking_load,
      params=metric_params,
    ),
    "allostasis/proxemic_stress": MetricsTermCfg(
      func=mdp.proxemic_stress,
      params=metric_params,
    ),
    "allostasis/motor_adaptation_cost": MetricsTermCfg(
      func=mdp.motor_adaptation_cost,
      params=metric_params,
    ),
    "allostasis/human_waiting_cost": MetricsTermCfg(
      func=mdp.human_waiting_cost,
      params=metric_params,
    ),
    "allostasis/human_reach_effort": MetricsTermCfg(
      func=mdp.human_reach_effort,
      params=metric_params,
    ),
  }

  cfg.viewer.body_name = "arm"
  cfg.viewer.distance = 1.25
  cfg.viewer.elevation = -12.0
  cfg.viewer.azimuth = 135.0
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.commands["handover"].resampling_time_range = (1000.0, 1000.0)

  return cfg


def allostatic_handover_full_yam_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Yam + HRGym-human full-fidelity Mjlab handover config."""
  cfg = allostatic_handover_yam_env_cfg(play=play)

  robot = cfg.scene.entities["robot"]
  # The upstream Yam lift task only enables gripper collisions, which is enough
  # for cube lifting but makes the visible base/arm pass through the HRGym table.
  # Full handover keeps the same robot asset while enabling all Yam collision
  # geoms so the tabletop scene has matching visual and physical geometry.
  robot.collisions = (YAM_FULL_COLLISION,)
  _set_yam_root_pose(robot, FULL_YAM_ROOT_POS)

  cfg.scene.entities = {
    name: entity
    for name, entity in cfg.scene.entities.items()
    if name not in {"cube", "human_torso", "human_upper_arm", "human_forearm", "human_hand"}
  }
  cfg.scene.entities.update(
    {
      "yam_pedestal": EntityCfg(
        spec_fn=_yam_pedestal_spec,
        init_state=EntityCfg.InitialStateCfg(
          pos=FULL_YAM_PEDESTAL_POS,
          joint_pos={},
        ),
      ),
      "table": EntityCfg(spec_fn=hrgym_table_spec),
      "human": EntityCfg(spec_fn=hrgym_human_spec),
      "manipulation_object": EntityCfg(
        spec_fn=hrgym_hammer_spec,
        init_state=EntityCfg.InitialStateCfg(
          pos=FULL_YAM_OBJECT_INIT_POS,
          rot=FULL_YAM_OBJECT_INIT_ROT,
          joint_pos={},
        ),
      ),
    }
  )
  cfg.scene.env_spacing = 2.4
  cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 256)
  cfg.sim.njmax = max(cfg.sim.njmax or 0, 1000)
  cfg.actions["arm_ik"] = mdp.TabletopDifferentialIKActionCfg(
    entity_name="robot",
    actuator_names=("joint[1-6]",),
    frame_type="site",
    frame_name="grasp_site",
    use_relative_mode=True,
    delta_pos_scale=0.045,
    orientation_weight=0.0,
    damping=0.055,
    max_dq=0.12,
    posture_weight=0.015,
    min_frame_z=FULL_YAM_MIN_GRASP_SITE_Z,
  )

  cfg.events["reset_table"] = EventTermCfg(
    func=velocity_mdp.reset_root_state_uniform,
    mode="reset",
    params={
      "pose_range": {},
      "velocity_range": {},
      "asset_cfg": SceneEntityCfg("table"),
    },
  )

  cfg.commands = {
    "handover": mdp.HrgymFullHandoverCommandCfg(
      resampling_time_range=(1000.0, 1000.0),
      debug_vis=True,
    )
  }
  command = cfg.commands["handover"]
  # Preserve the HRGym human root pose and animation. The reach-out segment
  # already enters Yam's tabletop workspace.
  command.human_base_pos_offset = FULL_YAM_HUMAN_BASE_POS_OFFSET
  # The HRGym object spawn range overlaps the Yam root after the tabletop
  # placement change. Spawn the hammer under Yam's forward workspace instead.
  command.object_pose_range.x = FULL_YAM_OBJECT_X_RANGE
  command.object_pose_range.y = FULL_YAM_OBJECT_Y_RANGE
  command.object_pose_range.z = (FULL_YAM_OBJECT_Z, FULL_YAM_OBJECT_Z)
  command.object_pose_range.yaw = FULL_YAM_OBJECT_YAW_RANGE
  command.robot_grasp_latch_enabled = True
  command.robot_grasp_distance_threshold = 0.16
  command.robot_grasp_action_threshold = 0.0
  command.handoff_object_offset = FULL_YAM_HANDOFF_OBJECT_OFFSET
  command.release_action_threshold = 0.15
  command.success_threshold = 0.23
  command.handoff_reach_progress_threshold = 0.1
  command.allow_release_away_from_hand = True
  command.start_with_object_grasped = False

  for obs_group in cfg.observations.values():
    if "ee_to_cube" in obs_group.terms:
      obs_group.terms["ee_to_cube"].params["object_name"] = "manipulation_object"
    if "cube_to_hand" in obs_group.terms:
      obs_group.terms["cube_to_hand"].params["object_name"] = "manipulation_object"

  for reward_name in ("handover", "handover_precise"):
    if reward_name in cfg.rewards:
      cfg.rewards[reward_name].params["object_name"] = "manipulation_object"

  cfg.rewards["robot_grasp_approach"] = RewardTermCfg(
    func=mdp.robot_grasp_approach_reward,
    weight=0.6,
    params={
      "command_name": "handover",
      "object_name": "manipulation_object",
      "std": 0.22,
      "asset_cfg": SceneEntityCfg("robot", site_names=("grasp_site",)),
    },
  )
  cfg.rewards["robot_grasp"] = RewardTermCfg(
    func=mdp.robot_grasp_bonus,
    weight=0.05,
    params={"command_name": "handover"},
  )
  cfg.rewards["carry_to_hand"] = RewardTermCfg(
    func=mdp.robot_carry_to_hand_reward,
    weight=0.25,
    params={"command_name": "handover", "std": 0.5},
  )
  cfg.rewards["release_at_hand"] = RewardTermCfg(
    func=mdp.release_at_hand_reward,
    weight=4.0,
    params={"command_name": "handover", "std": 0.35},
  )
  cfg.rewards["handoff"] = RewardTermCfg(
    func=mdp.handoff_bonus,
    weight=1.0,
    params={"command_name": "handover"},
  )

  metric_params = {"command_name": "handover"}
  cfg.metrics.update(
    {
      "animation/current_id": MetricsTermCfg(
        func=mdp.animation_current_id,
        params=metric_params,
        reduce="last",
      ),
      "animation/frame": MetricsTermCfg(
        func=mdp.animation_frame,
        params=metric_params,
        reduce="last",
      ),
      "handover/object_attached": MetricsTermCfg(
        func=mdp.object_attached,
        params=metric_params,
        reduce="last",
      ),
      "handover/palm_distance": MetricsTermCfg(
        func=mdp.palm_distance,
        params=metric_params,
        reduce="last",
      ),
      "handover/robot_object_grasped": MetricsTermCfg(
        func=mdp.robot_object_grasped,
        params=metric_params,
        reduce="last",
      ),
    }
  )

  cfg.viewer.origin_type = cfg.viewer.OriginType.WORLD
  cfg.viewer.entity_name = None
  cfg.viewer.body_name = None
  cfg.viewer.lookat = (0.90, 0.05, 0.90)
  cfg.viewer.distance = 3.8
  cfg.viewer.elevation = -30.0
  cfg.viewer.azimuth = 135.0
  return cfg


def allostatic_handover_full_task_only_yam_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a pure HRGym-layout tabletop grasp-and-transport task using Yam."""
  cfg = allostatic_handover_full_yam_env_cfg(play=play)
  cfg.actions.pop("speech", None)
  cfg.actions["arm_ik"].delta_pos_scale = 0.10
  cfg.actions["arm_ik"].max_dq = 0.22

  command = cfg.commands["handover"]
  command.reward_variant = "task_only"
  command.pure_task_mode = True
  command.require_readiness_for_reach = False
  command.readiness_initial = 1.0
  command.readiness_threshold = 0.0
  command.readiness_decay = 0.0
  command.readiness_load_sensitivity = 0.0
  command.overload_threshold = 1.0e9
  command.withdrawal_threshold = 1.0e9
  command.handoff_reach_progress_threshold = 0.1
  # Keep HRGym's walking animation, translated as a whole into Yam's reachable
  # workspace so the table-start task is physically learnable.
  command.animation_names = ("RobotHumanHandover/0",)
  command.human_base_pos_offset = FULL_YAM_HUMAN_BASE_POS_OFFSET
  command.robot_grasp_latch_enabled = True
  command.robot_grasp_distance_threshold = 0.16
  command.robot_grasp_action_threshold = 0.0
  command.release_action_threshold = 0.15
  command.success_threshold = 0.23
  command.min_lift_before_handoff = 0.08
  command.pure_task_min_distance_improvement = 0.0
  # Natural task reset: the object starts on the table and the policy must first
  # move to and grasp it. The old grasped-start shortcut is registered as a
  # separate curriculum task below for reproducing earlier checkpoints.
  command.start_with_object_grasped = False
  # Keep the task physically meaningful: opening the gripper away from the
  # receiving hand drops the object, so PPO must learn release timing.
  command.allow_release_away_from_hand = True
  command.freeze_hand_target_after_reset = False

  for group in cfg.observations.values():
    for term_name in (
      "readiness_belief",
      "load_proxy",
      "speech_context",
      "privileged_human_state",
      "privileged_load",
    ):
      group.terms.pop(term_name, None)

  for reward_name in ("speech_penalty", "allostatic_load", "waiting_cost"):
    cfg.rewards.pop(reward_name, None)

  cfg.rewards["handover"].weight = 0.0
  cfg.rewards["handover_precise"].weight = 0.4
  cfg.rewards["success"].weight = 30.0
  cfg.rewards["robot_grasp_approach"] = RewardTermCfg(
    func=mdp.robot_grasp_approach_reward,
    weight=1.2,
    params={
      "command_name": "handover",
      "object_name": "manipulation_object",
      "std": 0.26,
      "asset_cfg": SceneEntityCfg("robot", site_names=("grasp_site",)),
    },
  )
  cfg.rewards["robot_grasp"] = RewardTermCfg(
    func=mdp.robot_grasp_bonus,
    weight=0.15,
    params={"command_name": "handover"},
  )
  cfg.rewards["carry_to_hand"] = RewardTermCfg(
    func=mdp.robot_carry_to_hand_reward,
    weight=0.25,
    params={"command_name": "handover", "std": 0.5},
  )
  cfg.rewards["release_at_hand"] = RewardTermCfg(
    func=mdp.release_at_hand_reward,
    weight=5.0,
    params={"command_name": "handover", "std": 0.35},
  )
  cfg.rewards["handoff"] = RewardTermCfg(
    func=mdp.handoff_bonus,
    weight=25.0,
    params={"command_name": "handover"},
  )
  cfg.rewards["time_penalty"] = RewardTermCfg(
    func=mdp.time_penalty,
    weight=-0.02,
  )
  cfg.rewards["action_rate_l2"].weight = -0.002

  return cfg


def allostatic_handover_full_task_only_speech_yam_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a task-only reward condition where speech prepares the human.

  Human FSM/readiness/load are treated as hidden state for this comparison
  condition. They remain available through metrics/logs, but neither actor nor
  critic receives true hidden values or hand-coded load/readiness proxies.
  """
  cfg = allostatic_handover_full_yam_env_cfg(play=play)
  cfg.actions["arm_ik"].delta_pos_scale = 0.10
  cfg.actions["arm_ik"].max_dq = 0.22

  command = cfg.commands["handover"]
  command.reward_variant = "task_only"
  command.pure_task_mode = False
  command.require_readiness_for_reach = True
  command.require_readiness_for_animation_start = True
  command.readiness_initial = 0.24
  command.readiness_threshold = 0.62
  command.readiness_decay = 0.003
  command.readiness_load_sensitivity = 0.001
  command.announce_effect = 0.52
  command.ask_ready_effect = 0.18
  command.reassure_effect = 0.24
  command.waiting_effect = 0.06
  command.releasing_effect = 0.34
  command.confirmation_effect = 0.04
  command.repeated_effect_scale = 0.25
  command.readiness_hold_steps = 180
  command.readiness_hold_floor = 0.78
  # Load is hidden from the policy, but it still drives FSM transitions under
  # sustained high load. These thresholds are intentionally high enough that
  # ordinary adaptation does not immediately make handover impossible.
  command.overload_threshold = 7.0
  command.withdrawal_threshold = 9.0
  command.handoff_reach_progress_threshold = 0.1
  command.animation_names = ("RobotHumanHandover/0",)
  command.human_base_pos_offset = FULL_YAM_HUMAN_BASE_POS_OFFSET
  command.robot_grasp_latch_enabled = True
  command.robot_grasp_distance_threshold = 0.16
  command.robot_grasp_action_threshold = 0.0
  command.release_action_threshold = 0.15
  command.success_threshold = 0.23
  command.min_lift_before_handoff = 0.08
  command.pure_task_min_distance_improvement = 0.0
  command.start_with_object_grasped = False
  command.allow_release_away_from_hand = True
  command.freeze_hand_target_after_reset = False

  for reward_name in ("speech_penalty", "allostatic_load", "waiting_cost"):
    cfg.rewards.pop(reward_name, None)

  for group in cfg.observations.values():
    for term_name in (
      "readiness_belief",
      "load_proxy",
      "privileged_human_state",
      "privileged_load",
    ):
      group.terms.pop(term_name, None)

  cfg.rewards["handover"].weight = 0.0
  cfg.rewards["handover_precise"].weight = 0.4
  cfg.rewards["success"].weight = 30.0
  cfg.rewards["robot_grasp_approach"] = RewardTermCfg(
    func=mdp.robot_grasp_approach_reward,
    weight=1.2,
    params={
      "command_name": "handover",
      "object_name": "manipulation_object",
      "std": 0.26,
      "asset_cfg": SceneEntityCfg("robot", site_names=("grasp_site",)),
    },
  )
  cfg.rewards["robot_grasp"] = RewardTermCfg(
    func=mdp.robot_grasp_bonus,
    weight=0.15,
    params={"command_name": "handover"},
  )
  cfg.rewards["carry_to_hand"] = RewardTermCfg(
    func=mdp.robot_carry_to_hand_reward,
    weight=0.25,
    params={"command_name": "handover", "std": 0.5},
  )
  cfg.rewards["release_at_hand"] = RewardTermCfg(
    func=mdp.release_at_hand_reward,
    weight=5.0,
    params={"command_name": "handover", "std": 0.35},
  )
  cfg.rewards["handoff"] = RewardTermCfg(
    func=mdp.handoff_bonus,
    weight=25.0,
    params={"command_name": "handover"},
  )
  cfg.rewards["time_penalty"] = RewardTermCfg(
    func=mdp.time_penalty,
    weight=-0.02,
  )
  cfg.rewards["action_rate_l2"].weight = -0.002

  return cfg


def allostatic_handover_full_speech_penalty_yam_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the hidden-load speech-penalty comparison condition.

  This keeps the same 5D action, readiness FSM, observation space, object pose
  and task rewards as TaskOnlySpeech, but actor and critic do not observe the
  readiness/load proxies. Speech is penalized through a hidden accumulated
  speech load so sparse, well-timed speech remains cheap while sustained high
  frequency speech becomes smoothly more expensive.
  """
  cfg = allostatic_handover_full_task_only_speech_yam_env_cfg(play=play)
  command = cfg.commands["handover"]
  command.reward_variant = "speech_penalty"
  command.speech_penalty_load_threshold = 0.8
  command.speech_penalty_exp_scale = 0.5
  command.speech_penalty_max_excess = 4.0
  cfg.rewards["speech_penalty"] = RewardTermCfg(
    func=mdp.speech_penalty,
    weight=-0.04,
    params={"command_name": "handover"},
  )
  return cfg


def allostatic_handover_full_allostatic_belief_yam_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the learned-belief allostatic comparison condition.

  The policy keeps the same 5D motor+speech action space as TaskOnlySpeech and
  SpeechPenalty, but neither actor nor critic receives true hidden human state,
  true load, or hand-coded readiness/load proxies. Instead both receive outputs
  from a frozen world-model belief estimator loaded through
  ``ALLOSTATIC_WM_BELIEF_MODEL`` or the default
  ``outputs/world_model/latest/belief_distill.pt`` path.
  """
  cfg = allostatic_handover_full_task_only_speech_yam_env_cfg(play=play)
  command = cfg.commands["handover"]
  command.reward_variant = "allostatic"

  ee_cfg = SceneEntityCfg("robot", site_names=("grasp_site",))
  robot_joints_cfg = SceneEntityCfg("robot", joint_names=(".*",))
  wm_params = {
    "object_name": "manipulation_object",
    "command_name": "handover",
    "model_path": DEFAULT_BELIEF_MODEL_PATH,
    "asset_cfg": ee_cfg,
    "robot_joints_cfg": robot_joints_cfg,
    "belief_dim": 16,
    "num_human_states": 6,
  }
  for group in cfg.observations.values():
    for term_name in (
      "readiness_belief",
      "load_proxy",
      "privileged_human_state",
      "privileged_load",
    ):
      group.terms.pop(term_name, None)
    group.terms["wm_belief"] = ObservationTermCfg(
      func=mdp.wm_belief,
      params=wm_params,
    )
    group.terms["wm_human_state_probs"] = ObservationTermCfg(
      func=mdp.wm_human_state_probs,
      params=wm_params,
    )
    group.terms["wm_readiness_pred"] = ObservationTermCfg(
      func=mdp.wm_readiness_pred,
      params=wm_params,
    )
    group.terms["wm_load_pred"] = ObservationTermCfg(
      func=mdp.wm_load_pred,
      params=wm_params,
    )

  cfg.rewards["speech_penalty"] = RewardTermCfg(
    func=mdp.speech_penalty,
    weight=-0.02,
    params={"command_name": "handover"},
  )
  cfg.rewards["allostatic_load"] = RewardTermCfg(
    func=mdp.allostatic_load_penalty,
    weight=-0.05,
    params={"command_name": "handover"},
  )
  cfg.rewards["waiting_cost"] = RewardTermCfg(
    func=mdp.waiting_cost,
    weight=-0.10,
    params={"command_name": "handover"},
  )
  return cfg


def allostatic_handover_full_grasped_start_yam_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a 5D allostatic Full curriculum task with the object initially held."""
  cfg = allostatic_handover_full_yam_env_cfg(play=play)
  command = cfg.commands["handover"]
  command.start_with_object_grasped = True
  # During this curriculum stage, do not let exploratory opening drop the object
  # before the policy has learned to move it to the animated hand target.
  command.allow_release_away_from_hand = False
  # One meaningful handover cue should be enough to prepare the human for a
  # short window. Otherwise PPO learns that avoiding speech penalties is easier
  # than coordinating the allostatic handoff.
  command.readiness_initial = 0.42
  command.announce_effect = 0.46
  command.ask_ready_effect = 0.20
  command.reassure_effect = 0.30
  command.readiness_hold_steps = 150
  command.readiness_hold_floor = 0.74
  command.release_action_threshold = 0.0

  cfg.rewards["handover"].weight = 0.2
  cfg.rewards["handover_precise"].weight = 0.5
  cfg.rewards["success"].weight = 40.0
  cfg.rewards["robot_grasp"].weight = 0.0
  cfg.rewards["carry_to_hand"].weight = 0.6
  cfg.rewards["release_at_hand"].weight = 8.0
  cfg.rewards["release_intent_at_hand"] = RewardTermCfg(
    func=mdp.release_intent_at_hand_reward,
    weight=4.0,
    params={"command_name": "handover"},
  )
  cfg.rewards["handoff"].weight = 20.0
  cfg.rewards["speech_penalty"].weight = -0.02
  cfg.rewards["allostatic_load"].weight = -0.05
  cfg.rewards["waiting_cost"].weight = -0.2
  cfg.rewards["time_penalty"] = RewardTermCfg(
    func=mdp.time_penalty,
    weight=-0.02,
  )
  return cfg


def allostatic_handover_full_task_only_grasped_start_yam_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the elevated, object-in-gripper curriculum task used by old runs."""
  cfg = allostatic_handover_full_task_only_yam_env_cfg(play=play)
  _set_yam_root_pose(
    cfg.scene.entities["robot"],
    FULL_YAM_GRASPED_START_ROOT_POS,
    FULL_YAM_GRASPED_START_ROOT_QUAT,
  )

  pedestal = cfg.scene.entities["yam_pedestal"]
  pedestal.spec_fn = _yam_grasped_start_pedestal_spec
  pedestal.init_state.pos = FULL_YAM_GRASPED_START_PEDESTAL_POS

  command = cfg.commands["handover"]
  command.start_with_object_grasped = True
  return cfg
