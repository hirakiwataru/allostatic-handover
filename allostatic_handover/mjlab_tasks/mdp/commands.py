"""Vectorized allostatic handover command/state term for Mjlab."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch

from allostatic_handover.envs.human_hidden_state import HumanState
from allostatic_handover.envs.speech_events import RobotSpeechToken, speech_text
from allostatic_handover.mjlab_tasks.hrgym_assets import (
  DEFAULT_FULL_ANIMATION_NAMES,
  DEFAULT_VENDOR_ROOT,
  HRGYM_HUMAN_JOINT_NAMES,
  HrgymAnimationLibrary,
  xyzw_to_wxyz,
)
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_from_euler_xyz,
  quat_mul,
  sample_uniform,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class HandoverPhase(IntEnum):
  APPROACH = 0
  REACH_OUT = 1
  RETREAT = 2
  COMPLETE = 3


def speech_tokens_from_scalar(scalar: torch.Tensor) -> torch.Tensor:
  """Vectorized equivalent of robot_speech_from_scalar."""
  clipped = torch.clamp(scalar, -1.0, 1.0)
  max_idx = len(RobotSpeechToken) - 1
  return torch.round((clipped + 1.0) * 0.5 * max_idx).long()


def _to_float(value: torch.Tensor) -> float:
  return float(value.detach().float().cpu().item())


def _to_int(value: torch.Tensor) -> int:
  return int(value.detach().cpu().item())


def _to_bool(value: torch.Tensor) -> bool:
  return bool(value.detach().bool().cpu().item())


@dataclass(kw_only=True)
class AllostaticHandoverCommandCfg(CommandTermCfg):
  """State and target generator for the Mjlab allostatic handover task."""

  entity_name: str = "cube"
  robot_entity_name: str = "robot"
  hand_entity_name: str | None = "human_hand"
  torso_entity_name: str | None = "human_torso"
  upper_arm_entity_name: str | None = "human_upper_arm"
  forearm_entity_name: str | None = "human_forearm"
  speech_action_name: str = "speech"
  gripper_action_name: str = "gripper"
  reward_variant: Literal["task_only", "speech_penalty", "allostatic"] = "allostatic"
  pure_task_mode: bool = False

  readiness_initial: float = 0.28
  readiness_threshold: float = 0.65
  readiness_decay: float = 0.004
  readiness_hold_steps: int = 90
  readiness_hold_floor: float = 0.72
  readiness_load_sensitivity: float = 0.006

  announce_effect: float = 0.34
  ask_ready_effect: float = 0.16
  reassure_effect: float = 0.24
  waiting_effect: float = 0.08
  releasing_effect: float = 0.28
  confirmation_effect: float = 0.04
  repeated_effect_scale: float = 0.35

  approach_min_steps: int = 8
  reach_steps: int = 55
  retreat_steps: int = 65
  success_threshold: float = 0.075
  release_action_threshold: float = 0.15
  robot_grasp_latch_enabled: bool = False
  robot_grasp_distance_threshold: float = 0.065
  robot_grasp_action_threshold: float = -0.15
  robot_grasp_object_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
  handoff_object_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
  start_with_object_grasped: bool = False
  allow_release_away_from_hand: bool = True
  pure_task_min_distance_improvement: float = 0.0
  min_lift_before_handoff: float = 0.0

  load_decay: float = 0.975
  load_recovery: float = 0.008
  speech_load_cost: float = 0.035
  ask_ready_load_cost: float = 0.055
  repeated_speech_load_cost: float = 0.11
  speech_penalty_load_threshold: float = 0.8
  speech_penalty_exp_scale: float = 0.5
  speech_penalty_max_excess: float = 4.0
  withdrawal_recovery_steps: int = 120
  withdrawal_recovery_rate: float = 0.08
  withdrawal_recovery_decay: float = 0.02
  withdrawal_recovery_load_relief: float = 4.0
  reach_effort_cost: float = 0.22
  close_distance: float = 0.18
  proxemic_stress_gain: float = 0.65
  overload_threshold: float = 7.0
  withdrawal_threshold: float = 9.0

  @dataclass
  class ObjectPoseRangeCfg:
    x: tuple[float, float] = (0.26, 0.38)
    y: tuple[float, float] = (-0.14, 0.14)
    z: tuple[float, float] = (0.035, 0.055)
    yaw: tuple[float, float] = (-3.14, 3.14)

  object_pose_range: ObjectPoseRangeCfg = field(default_factory=ObjectPoseRangeCfg)

  @dataclass
  class HumanPoseCfg:
    torso: tuple[float, float, float] = (0.52, -0.55, 0.55)
    shoulder: tuple[float, float, float] = (0.48, -0.43, 0.46)
    rest_hand: tuple[float, float, float] = (0.50, -0.34, 0.34)
    reach_hand: tuple[float, float, float] = (0.40, -0.05, 0.28)
    retreat_hand: tuple[float, float, float] = (0.54, -0.43, 0.38)

  human_pose: HumanPoseCfg = field(default_factory=HumanPoseCfg)

  def build(self, env: ManagerBasedRlEnv) -> AllostaticHandoverCommand:
    return AllostaticHandoverCommand(self, env)


@dataclass(kw_only=True)
class HrgymFullHandoverCommandCfg(AllostaticHandoverCommandCfg):
  """HRGym-style handover command using the vendor-copied human animation data."""

  entity_name: str = "manipulation_object"
  hand_entity_name: str | None = None
  torso_entity_name: str | None = None
  upper_arm_entity_name: str | None = None
  forearm_entity_name: str | None = None
  human_entity_name: str = "human"
  animation_vendor_root: str = str(DEFAULT_VENDOR_ROOT)
  animation_names: tuple[str, ...] = DEFAULT_FULL_ANIMATION_NAMES
  human_animation_freq: float = 90.0
  human_base_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
  # HRGym uses Rotation.from_quat([0.5, 0.5, 0.5, 0.5]) as the base transform
  # from animation coordinates into MuJoCo world coordinates. Stored here as wxyz.
  human_base_quat_wxyz: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 0.5)
  require_readiness_for_reach: bool = True
  require_readiness_for_animation_start: bool = False
  handoff_reach_progress_threshold: float = 0.25
  freeze_hand_target_after_reset: bool = False
  ready_loop_amplitude_scale: float = 0.25
  hesitant_loop_amplitude_scale: float = 1.0

  @dataclass
  class ObjectPoseRangeCfg:
    x: tuple[float, float] = (0.315, 0.525)
    y: tuple[float, float] = (-0.1425, 0.1425)
    z: tuple[float, float] = (0.97084304, 0.97084304)
    yaw: tuple[float, float] = (0.0, 0.0)

  object_pose_range: ObjectPoseRangeCfg = field(default_factory=ObjectPoseRangeCfg)

  def build(self, env: ManagerBasedRlEnv) -> HrgymFullHandoverCommand:
    return HrgymFullHandoverCommand(self, env)


class AllostaticHandoverCommand(CommandTerm):
  """Tracks hand target, hidden readiness, allostatic load and handover phase."""

  cfg: AllostaticHandoverCommandCfg

  def __init__(self, cfg: AllostaticHandoverCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.object: Entity = env.scene[cfg.entity_name]
    self.robot: Entity = env.scene[cfg.robot_entity_name]
    self.hand: Entity | None = (
      env.scene[cfg.hand_entity_name] if cfg.hand_entity_name is not None else None
    )
    self.torso: Entity | None = (
      env.scene[cfg.torso_entity_name] if cfg.torso_entity_name is not None else None
    )
    self.upper_arm: Entity | None = (
      env.scene[cfg.upper_arm_entity_name]
      if cfg.upper_arm_entity_name is not None
      else None
    )
    self.forearm: Entity | None = (
      env.scene[cfg.forearm_entity_name]
      if cfg.forearm_entity_name is not None
      else None
    )
    site_ids, _ = self.robot.find_sites(("grasp_site",))
    self._grasp_site_id = site_ids[0]

    self._env_ids_all = torch.arange(self.num_envs, device=self.device)
    self._identity_quat = torch.tensor(
      [1.0, 0.0, 0.0, 0.0], device=self.device
    ).repeat(self.num_envs, 1)
    self._zero_velocity = torch.zeros(self.num_envs, 6, device=self.device)
    self._last_update_step = -1

    self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
    self.hand_pos = torch.zeros(self.num_envs, 3, device=self.device)
    self.phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.phase_step = torch.zeros(self.num_envs, device=self.device)
    self.reach_progress = torch.zeros(self.num_envs, device=self.device)
    self.retreat_progress = torch.zeros(self.num_envs, device=self.device)
    self.episode_success = torch.zeros(self.num_envs, device=self.device)
    self.object_attached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.robot_object_grasped = torch.zeros(
      self.num_envs,
      dtype=torch.bool,
      device=self.device,
    )
    self.initial_object_z = torch.zeros(self.num_envs, device=self.device)
    self.max_object_lift = torch.zeros(self.num_envs, device=self.device)
    self.release_at_hand_event = torch.zeros(self.num_envs, device=self.device)
    self.last_handoff_ready_gate = torch.zeros(
      self.num_envs,
      dtype=torch.bool,
      device=self.device,
    )
    self.last_handoff_reaching_gate = torch.zeros(
      self.num_envs,
      dtype=torch.bool,
      device=self.device,
    )
    self.last_handoff_reach_progress_gate = torch.zeros(
      self.num_envs,
      dtype=torch.bool,
      device=self.device,
    )
    self.last_handoff_distance_gate = torch.zeros(
      self.num_envs,
      dtype=torch.bool,
      device=self.device,
    )
    self.last_handoff_release_gate = torch.zeros(
      self.num_envs,
      dtype=torch.bool,
      device=self.device,
    )
    self.last_handoff_lift_gate = torch.ones(
      self.num_envs,
      dtype=torch.bool,
      device=self.device,
    )
    self.last_handoff_grasp_gate = torch.ones(
      self.num_envs,
      dtype=torch.bool,
      device=self.device,
    )
    self.last_handoff_now = torch.zeros(
      self.num_envs,
      dtype=torch.bool,
      device=self.device,
    )

    self.human_readiness = torch.zeros(self.num_envs, device=self.device)
    self.readiness_belief = torch.zeros(self.num_envs, device=self.device)
    self.readiness_hold = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.withdrawal_recovery = torch.zeros(self.num_envs, device=self.device)
    self.withdrawal_recovery_hold = torch.zeros(
      self.num_envs,
      dtype=torch.long,
      device=self.device,
    )
    self.human_state_id = torch.full(
      (self.num_envs,),
      int(HumanState.HESITANT),
      dtype=torch.long,
      device=self.device,
    )

    self.last_speech_token = torch.full(
      (self.num_envs,),
      int(RobotSpeechToken.SILENCE),
      dtype=torch.long,
      device=self.device,
    )
    self.previous_speech_token = self.last_speech_token.clone()
    self.current_speech_is_silence = torch.ones(self.num_envs, device=self.device)
    self.current_speech_is_repeated = torch.zeros(self.num_envs, device=self.device)
    self.current_speech_is_active = torch.zeros(self.num_envs, device=self.device)

    self.robot_speech_count = torch.zeros(self.num_envs, device=self.device)
    self.silence_count = torch.zeros(self.num_envs, device=self.device)
    self.repeated_speech_count = torch.zeros(self.num_envs, device=self.device)
    self.handoff_count = torch.zeros(self.num_envs, device=self.device)

    self.attention_load = torch.zeros(self.num_envs, device=self.device)
    self.turn_taking_load = torch.zeros(self.num_envs, device=self.device)
    self.proxemic_stress = torch.zeros(self.num_envs, device=self.device)
    self.motor_adaptation_cost = torch.zeros(self.num_envs, device=self.device)
    self.human_waiting_cost = torch.zeros(self.num_envs, device=self.device)
    self.human_reach_effort = torch.zeros(self.num_envs, device=self.device)
    self.allostatic_load_total = torch.zeros(self.num_envs, device=self.device)

    for name in (
      "success",
      "robot_speech_count",
      "silence_ratio",
      "repeated_speech_count",
      "human_readiness",
      "human_reach_progress",
      "allostatic_load_total",
      "attention_load",
      "turn_taking_load",
      "proxemic_stress",
      "motor_adaptation_cost",
      "human_waiting_cost",
      "human_reach_effort",
      "robot_object_grasped",
    ):
      self.metrics[name] = torch.zeros(self.num_envs, device=self.device)
    self._robot_grasp_object_offset = torch.tensor(
      cfg.robot_grasp_object_offset,
      dtype=torch.float32,
      device=self.device,
    )
    self._handoff_object_offset = torch.tensor(
      cfg.handoff_object_offset,
      dtype=torch.float32,
      device=self.device,
    )
    self._live_log_path = os.environ.get("ALLOSTATIC_MJLAB_LIVE_LOG")
    self._live_log_interval = max(
      1,
      int(os.environ.get("ALLOSTATIC_MJLAB_LIVE_LOG_INTERVAL", "10")),
    )
    self._live_log_env_idx = int(os.environ.get("ALLOSTATIC_MJLAB_LIVE_ENV", "0"))
    self._live_log_last_step = -1
    if self._live_log_path:
      path = Path(self._live_log_path)
      path.parent.mkdir(parents=True, exist_ok=True)
      path.write_text("", encoding="utf-8")

  @property
  def command(self) -> torch.Tensor:
    return self.target_pos

  def pre_reward_update(self) -> None:
    """Synchronize latent state before rewards, metrics, observations and dones."""
    step = int(self._env.common_step_counter)
    if self._last_update_step == step:
      return
    self._last_update_step = step

    scalar = self._speech_scalar()
    token = speech_tokens_from_scalar(scalar)
    self.previous_speech_token[:] = self.last_speech_token
    self.last_speech_token[:] = token

    active = token != int(RobotSpeechToken.SILENCE)
    repeated = active & (token == self.previous_speech_token)
    silence = ~active
    self.current_speech_is_active[:] = active.float()
    self.current_speech_is_silence[:] = silence.float()
    self.current_speech_is_repeated[:] = repeated.float()
    self.robot_speech_count += active.float()
    self.silence_count += silence.float()
    self.repeated_speech_count += repeated.float()

    self._update_readiness(token, active, repeated)
    self._update_phase_and_handoff()
    self._update_load(active, repeated, token)
    self._update_human_state()
    self._write_visual_state()
    self._update_episode_metrics()
    self._write_live_log()

  def _speech_scalar(self) -> torch.Tensor:
    if self.cfg.speech_action_name not in self._env.action_manager.active_terms:
      return torch.full(
        (self.num_envs,),
        -1.0,
        device=self.device,
      )
    term = self._env.action_manager.get_term(self.cfg.speech_action_name)
    return term.raw_action[:, 0]

  def _gripper_release_signal(self) -> torch.Tensor:
    if self.cfg.gripper_action_name not in self._env.action_manager.active_terms:
      return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
    term = self._env.action_manager.get_term(self.cfg.gripper_action_name)
    return term.raw_action[:, 0] > self.cfg.release_action_threshold

  def _gripper_close_signal(self) -> torch.Tensor:
    if self.cfg.gripper_action_name not in self._env.action_manager.active_terms:
      return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    term = self._env.action_manager.get_term(self.cfg.gripper_action_name)
    return term.raw_action[:, 0] < self.cfg.robot_grasp_action_threshold

  def _robot_ee_pos(self) -> torch.Tensor:
    return self.robot.data.site_pos_w[:, self._grasp_site_id, :]

  def _sync_robot_grasped_object(self, object_pos: torch.Tensor) -> torch.Tensor:
    if not self.cfg.robot_grasp_latch_enabled:
      return object_pos

    ee_pos = self._robot_ee_pos()
    distance = torch.norm(object_pos - ee_pos, dim=-1)
    grasp_now = (
      ~self.object_attached
      & self._gripper_close_signal()
      & (distance <= self.cfg.robot_grasp_distance_threshold)
    )
    self.robot_object_grasped |= grasp_now
    held = self.robot_object_grasped & ~self.object_attached
    held_pos = ee_pos + self._robot_grasp_object_offset
    if held.any():
      held_ids = held.nonzero().flatten()
      self._write_entity_pose(self.object, held_pos[held_ids], held_ids)
    return torch.where(held.unsqueeze(-1), held_pos, object_pos)

  def _start_with_robot_grasped_object(self, env_ids: torch.Tensor) -> None:
    if not (self.cfg.robot_grasp_latch_enabled and self.cfg.start_with_object_grasped):
      return
    ee_pos = self._robot_ee_pos()[env_ids] + self._robot_grasp_object_offset
    self.robot_object_grasped[env_ids] = True
    self._write_entity_pose(self.object, ee_pos, env_ids)

  def _handoff_reference_pos(self, object_pos: torch.Tensor) -> torch.Tensor:
    held_by_robot = self.robot_object_grasped & ~self.object_attached
    return torch.where(
      held_by_robot.unsqueeze(-1),
      object_pos + self._handoff_object_offset,
      object_pos,
    )

  def _speech_readiness_effect(self, token: torch.Tensor) -> torch.Tensor:
    effect = torch.zeros(self.num_envs, device=self.device)
    effect = torch.where(
      token == int(RobotSpeechToken.ANNOUNCE_HANDOVER),
      torch.full_like(effect, self.cfg.announce_effect),
      effect,
    )
    effect = torch.where(
      token == int(RobotSpeechToken.ASK_READY),
      torch.full_like(effect, self.cfg.ask_ready_effect),
      effect,
    )
    effect = torch.where(
      token == int(RobotSpeechToken.REASSURE),
      torch.full_like(effect, self.cfg.reassure_effect),
      effect,
    )
    effect = torch.where(
      token == int(RobotSpeechToken.SAY_WAITING),
      torch.full_like(effect, self.cfg.waiting_effect),
      effect,
    )
    effect = torch.where(
      token == int(RobotSpeechToken.SAY_RELEASING),
      torch.full_like(effect, self.cfg.releasing_effect),
      effect,
    )
    effect = torch.where(
      token == int(RobotSpeechToken.ASK_CONFIRMATION),
      torch.full_like(effect, self.cfg.confirmation_effect),
      effect,
    )
    return effect

  def _update_readiness(
    self,
    token: torch.Tensor,
    active: torch.Tensor,
    repeated: torch.Tensor,
  ) -> None:
    if self.cfg.pure_task_mode:
      self.human_readiness[:] = 1.0
      self.readiness_belief[:] = 1.0
      self.readiness_hold[:] = 0
      return

    effect = self._speech_readiness_effect(token)
    effect = torch.where(repeated, effect * self.cfg.repeated_effect_scale, effect)
    self.human_readiness = torch.clamp(self.human_readiness + effect, 0.0, 1.0)

    hold_cue = (
      (token == int(RobotSpeechToken.ANNOUNCE_HANDOVER))
      | (token == int(RobotSpeechToken.SAY_RELEASING))
      | (token == int(RobotSpeechToken.REASSURE))
    )
    self.readiness_hold = torch.where(
      hold_cue,
      torch.full_like(self.readiness_hold, self.cfg.readiness_hold_steps),
      self.readiness_hold,
    )

    handover_intent_cue = token == int(RobotSpeechToken.ANNOUNCE_HANDOVER)
    withdrawal_like = (
      (self.human_state_id == int(HumanState.WITHDRAWING))
      | (self.allostatic_load_total >= self.cfg.withdrawal_threshold)
    )
    start_withdrawal_recovery = handover_intent_cue & withdrawal_like
    self.withdrawal_recovery_hold = torch.where(
      start_withdrawal_recovery,
      torch.full_like(self.withdrawal_recovery_hold, self.cfg.withdrawal_recovery_steps),
      self.withdrawal_recovery_hold,
    )
    recovery_active = self.withdrawal_recovery_hold > 0
    self.withdrawal_recovery_hold = torch.clamp(
      self.withdrawal_recovery_hold - 1,
      min=0,
    )
    self.withdrawal_recovery = torch.where(
      recovery_active,
      self.withdrawal_recovery
      + self.cfg.withdrawal_recovery_rate * (1.0 - self.withdrawal_recovery),
      self.withdrawal_recovery * (1.0 - self.cfg.withdrawal_recovery_decay),
    )
    self.withdrawal_recovery = torch.clamp(self.withdrawal_recovery, 0.0, 1.0)

    in_hold = self.readiness_hold > 0
    self.readiness_hold = torch.clamp(self.readiness_hold - 1, min=0)
    decay = torch.full_like(self.human_readiness, self.cfg.readiness_decay)
    decay = torch.where(active, decay * 0.25, decay)
    decay += self.allostatic_load_total * self.cfg.readiness_load_sensitivity
    self.human_readiness = torch.where(
      in_hold,
      torch.maximum(
        self.human_readiness - decay * 0.15,
        torch.full_like(self.human_readiness, self.cfg.readiness_hold_floor),
      ),
      self.human_readiness - decay,
    )
    self.human_readiness = torch.clamp(self.human_readiness, 0.0, 1.0)

    # A deliberately imperfect public belief: the robot can infer trends but not
    # observe hidden readiness exactly.
    self.readiness_belief = torch.clamp(
      0.92 * self.readiness_belief + 0.08 * self.human_readiness + 0.05 * active.float(),
      0.0,
      1.0,
    )

  def _update_phase_and_handoff(self) -> None:
    ready = self.human_readiness >= self.cfg.readiness_threshold
    phase = self.phase
    self.phase_step += 1.0

    can_reach = (
      (phase == int(HandoverPhase.APPROACH))
      & ready
      & (self.phase_step >= self.cfg.approach_min_steps)
    )
    self.phase = torch.where(
      can_reach,
      torch.full_like(self.phase, int(HandoverPhase.REACH_OUT)),
      self.phase,
    )
    self.phase_step = torch.where(can_reach, torch.zeros_like(self.phase_step), self.phase_step)

    reaching = self.phase == int(HandoverPhase.REACH_OUT)
    reach_delta = 1.0 / max(float(self.cfg.reach_steps), 1.0)
    self.reach_progress = torch.where(
      reaching & ready,
      torch.clamp(self.reach_progress + reach_delta, 0.0, 1.0),
      self.reach_progress,
    )
    self.reach_progress = torch.where(
      reaching & ~ready,
      torch.clamp(self.reach_progress - reach_delta * 0.25, 0.0, 1.0),
      self.reach_progress,
    )

    self.hand_pos[:] = self._interpolate_hand_position()
    self.target_pos[:] = self.hand_pos

    cube_pos = self._sync_robot_grasped_object(self.object.data.root_link_pos_w)
    self._update_object_lift(cube_pos)
    handoff_pos = self._handoff_reference_pos(cube_pos)
    cube_to_hand = torch.norm(handoff_pos - self.hand_pos, dim=-1)
    can_release = self._gripper_release_signal()
    self.release_at_hand_event = (
      self.robot_object_grasped.float()
      * can_release.float()
      * (cube_to_hand <= self.cfg.success_threshold * 1.5).float()
    )
    handoff_now = (
      reaching
      & (self.reach_progress >= 0.65)
      & ready
      & (cube_to_hand <= self.cfg.success_threshold)
      & can_release
    )
    lift_gate = torch.ones_like(handoff_now)
    if self.cfg.min_lift_before_handoff > 0.0:
      lift_gate = self.max_object_lift >= self.cfg.min_lift_before_handoff
      handoff_now &= lift_gate
    grasp_gate = torch.ones_like(handoff_now)
    if self.cfg.robot_grasp_latch_enabled:
      grasp_gate = self.robot_object_grasped
      handoff_now &= grasp_gate
    self._set_handoff_gate_diagnostics(
      ready=ready,
      reaching=reaching,
      reach_progress_gate=self.reach_progress >= 0.65,
      distance_gate=cube_to_hand <= self.cfg.success_threshold,
      release_gate=can_release,
      lift_gate=lift_gate,
      grasp_gate=grasp_gate,
      handoff_now=handoff_now,
    )
    newly_attached = handoff_now & ~self.object_attached
    self.object_attached |= handoff_now
    self.handoff_count += newly_attached.float()
    release_away = can_release & ~handoff_now
    if not self.cfg.allow_release_away_from_hand:
      release_away &= torch.zeros_like(release_away)
    self.robot_object_grasped &= ~(self.object_attached | release_away)

    to_retreat = self.object_attached & (self.phase == int(HandoverPhase.REACH_OUT))
    self.phase = torch.where(
      to_retreat,
      torch.full_like(self.phase, int(HandoverPhase.RETREAT)),
      self.phase,
    )
    self.phase_step = torch.where(to_retreat, torch.zeros_like(self.phase_step), self.phase_step)

    retreating = self.phase == int(HandoverPhase.RETREAT)
    retreat_delta = 1.0 / max(float(self.cfg.retreat_steps), 1.0)
    self.retreat_progress = torch.where(
      retreating,
      torch.clamp(self.retreat_progress + retreat_delta, 0.0, 1.0),
      self.retreat_progress,
    )
    self.hand_pos[:] = self._interpolate_hand_position()
    self.target_pos[:] = self.hand_pos

    complete = retreating & (self.retreat_progress >= 1.0)
    self.phase = torch.where(
      complete,
      torch.full_like(self.phase, int(HandoverPhase.COMPLETE)),
      self.phase,
    )
    self.episode_success = torch.maximum(self.episode_success, complete.float())

    if self.object_attached.any():
      attached_ids = self.object_attached.nonzero().flatten()
      self._write_entity_pose(self.object, self.hand_pos[attached_ids], attached_ids)

  def _set_handoff_gate_diagnostics(
    self,
    *,
    ready: torch.Tensor,
    reaching: torch.Tensor,
    reach_progress_gate: torch.Tensor,
    distance_gate: torch.Tensor,
    release_gate: torch.Tensor,
    lift_gate: torch.Tensor,
    grasp_gate: torch.Tensor,
    handoff_now: torch.Tensor,
  ) -> None:
    self.last_handoff_ready_gate[:] = ready
    self.last_handoff_reaching_gate[:] = reaching
    self.last_handoff_reach_progress_gate[:] = reach_progress_gate
    self.last_handoff_distance_gate[:] = distance_gate
    self.last_handoff_release_gate[:] = release_gate
    self.last_handoff_lift_gate[:] = lift_gate
    self.last_handoff_grasp_gate[:] = grasp_gate
    self.last_handoff_now[:] = handoff_now

  def _write_live_log(self) -> None:
    if not self._live_log_path:
      return
    step = int(self._env.common_step_counter)
    if step == self._live_log_last_step or step % self._live_log_interval != 0:
      return
    self._live_log_last_step = step
    idx = max(0, min(self._live_log_env_idx, self.num_envs - 1))
    token = RobotSpeechToken(int(self.last_speech_token[idx].detach().cpu().item()))
    state = HumanState(int(self.human_state_id[idx].detach().cpu().item()))
    payload = {
      "step": step,
      "episode_step": _to_int(self._env.episode_length_buf[idx]),
      "env": idx,
      "robot_speech_token": token.name,
      "robot_speech_text": speech_text(token),
      "speech_scalar": _to_float(self._speech_scalar()[idx]),
      "human_state": state.name,
      "phase": HandoverPhase(int(self.phase[idx].detach().cpu().item())).name,
      "human_readiness": _to_float(self.human_readiness[idx]),
      "readiness_belief": _to_float(self.readiness_belief[idx]),
      "readiness_hold": _to_int(self.readiness_hold[idx]),
      "allostatic_load_total": _to_float(self.allostatic_load_total[idx]),
      "attention_load": _to_float(self.attention_load[idx]),
      "turn_taking_load": _to_float(self.turn_taking_load[idx]),
      "proxemic_stress": _to_float(self.proxemic_stress[idx]),
      "human_waiting_cost": _to_float(self.human_waiting_cost[idx]),
      "human_reach_effort": _to_float(self.human_reach_effort[idx]),
      "reach_progress": _to_float(self.reach_progress[idx]),
      "retreat_progress": _to_float(self.retreat_progress[idx]),
      "robot_object_grasped": _to_bool(self.robot_object_grasped[idx]),
      "object_attached": _to_bool(self.object_attached[idx]),
      "episode_success": _to_float(self.episode_success[idx]),
      "palm_distance": _to_float(getattr(self, "palm_distance", torch.zeros_like(self.human_readiness))[idx]),
      "best_palm_distance": _to_float(getattr(self, "best_palm_distance", torch.zeros_like(self.human_readiness))[idx]),
      "success_threshold": float(self.cfg.success_threshold),
      "max_object_lift": _to_float(self.max_object_lift[idx]),
      "min_lift_before_handoff": float(self.cfg.min_lift_before_handoff),
      "handoff_count": _to_float(self.handoff_count[idx]),
      "gates": {
        "ready": _to_bool(self.last_handoff_ready_gate[idx]),
        "reaching": _to_bool(self.last_handoff_reaching_gate[idx]),
        "reach_progress": _to_bool(self.last_handoff_reach_progress_gate[idx]),
        "distance": _to_bool(self.last_handoff_distance_gate[idx]),
        "release": _to_bool(self.last_handoff_release_gate[idx]),
        "lift": _to_bool(self.last_handoff_lift_gate[idx]),
        "grasp": _to_bool(self.last_handoff_grasp_gate[idx]),
        "handoff_now": _to_bool(self.last_handoff_now[idx]),
      },
    }
    with Path(self._live_log_path).open("a", encoding="utf-8") as file:
      file.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

  def _interpolate_hand_position(self) -> torch.Tensor:
    origin = self._env.scene.env_origins
    hp = self.cfg.human_pose
    rest = torch.tensor(hp.rest_hand, device=self.device).expand(self.num_envs, 3) + origin
    reach = torch.tensor(hp.reach_hand, device=self.device).expand(self.num_envs, 3) + origin
    retreat = torch.tensor(hp.retreat_hand, device=self.device).expand(self.num_envs, 3) + origin

    reach_blend = self.reach_progress.unsqueeze(-1)
    hand = rest + (reach - rest) * reach_blend
    retreat_blend = self.retreat_progress.unsqueeze(-1)
    hand = torch.where(
      (self.phase == int(HandoverPhase.RETREAT)).unsqueeze(-1)
      | (self.phase == int(HandoverPhase.COMPLETE)).unsqueeze(-1),
      reach + (retreat - reach) * retreat_blend,
      hand,
    )
    return hand

  def _update_load(
    self,
    active: torch.Tensor,
    repeated: torch.Tensor,
    token: torch.Tensor,
  ) -> None:
    if self.cfg.pure_task_mode:
      for tensor in (
        self.attention_load,
        self.turn_taking_load,
        self.proxemic_stress,
        self.motor_adaptation_cost,
        self.human_waiting_cost,
        self.human_reach_effort,
        self.allostatic_load_total,
      ):
        tensor[:] = 0.0
      return

    ee_pos = self.robot.data.site_pos_w[:, self._grasp_site_id, :]
    ee_to_hand = torch.norm(ee_pos - self.hand_pos, dim=-1)
    close_excess = torch.clamp(self.cfg.close_distance - ee_to_hand, min=0.0)

    self.attention_load = (
      self.attention_load * self.cfg.load_decay
      + active.float() * self.cfg.speech_load_cost
      + (token == int(RobotSpeechToken.ASK_READY)).float() * self.cfg.ask_ready_load_cost
      - self.cfg.load_recovery
    ).clamp_min(0.0)
    self.turn_taking_load = (
      self.turn_taking_load * self.cfg.load_decay
      + repeated.float() * self.cfg.repeated_speech_load_cost
      - self.cfg.load_recovery
    ).clamp_min(0.0)
    self.proxemic_stress = (
      self.proxemic_stress * self.cfg.load_decay
      + close_excess * self.cfg.proxemic_stress_gain
      - self.cfg.load_recovery
    ).clamp_min(0.0)
    self.human_reach_effort = self.reach_progress * self.cfg.reach_effort_cost
    self.human_waiting_cost = (
      (self.phase == int(HandoverPhase.REACH_OUT)).float()
      * torch.clamp(1.0 - self.reach_progress, 0.0, 1.0)
      * 0.08
    )
    self.motor_adaptation_cost = 0.25 * torch.abs(self.human_readiness - self.readiness_belief)
    self.allostatic_load_total = (
      self.attention_load
      + self.turn_taking_load
      + self.proxemic_stress
      + self.motor_adaptation_cost
      + self.human_waiting_cost
      + self.human_reach_effort
    )

  def _update_human_state(self) -> None:
    if self.cfg.pure_task_mode:
      state = torch.full_like(self.human_state_id, int(HumanState.READY))
      state = torch.where(
        self.reach_progress > 0.05,
        torch.full_like(state, int(HumanState.GRASPING)),
        state,
      )
      self.human_state_id[:] = state
      return

    ready = self.human_readiness >= self.cfg.readiness_threshold
    state = torch.full_like(self.human_state_id, int(HumanState.HESITANT))
    state = torch.where(ready, torch.full_like(state, int(HumanState.READY)), state)
    state = torch.where(
      self.reach_progress > 0.05,
      torch.full_like(state, int(HumanState.GRASPING)),
      state,
    )
    effective_load_for_state = torch.clamp(
      self.allostatic_load_total
      - self.withdrawal_recovery * self.cfg.withdrawal_recovery_load_relief,
      min=0.0,
    )
    state = torch.where(
      effective_load_for_state >= self.cfg.overload_threshold,
      torch.full_like(state, int(HumanState.OVERLOADED)),
      state,
    )
    state = torch.where(
      effective_load_for_state >= self.cfg.withdrawal_threshold,
      torch.full_like(state, int(HumanState.WITHDRAWING)),
      state,
    )
    self.human_state_id[:] = state

  def _write_visual_state(self) -> None:
    if (
      self.hand is None
      or self.torso is None
      or self.upper_arm is None
      or self.forearm is None
    ):
      return
    origin = self._env.scene.env_origins
    hp = self.cfg.human_pose
    torso_pos = torch.tensor(hp.torso, device=self.device).expand(self.num_envs, 3) + origin
    shoulder = torch.tensor(hp.shoulder, device=self.device).expand(self.num_envs, 3) + origin
    elbow = shoulder + 0.55 * (self.hand_pos - shoulder)
    upper_pos = 0.5 * (shoulder + elbow)
    forearm_pos = 0.5 * (elbow + self.hand_pos)
    self._write_entity_pose(self.torso, torso_pos, None)
    self._write_entity_pose(self.upper_arm, upper_pos, None)
    self._write_entity_pose(self.forearm, forearm_pos, None)
    self._write_entity_pose(self.hand, self.hand_pos, None)

  def _write_entity_pose(
    self,
    entity: Entity,
    pos: torch.Tensor,
    env_ids: torch.Tensor | None,
  ) -> None:
    n = pos.shape[0]
    quat = self._identity_quat[:n]
    pose = torch.cat([pos, quat], dim=-1)
    velocity = self._zero_velocity[:n]
    entity.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    entity.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)

  def _update_episode_metrics(self) -> None:
    steps = torch.clamp(self._env.episode_length_buf.float() + 1.0, min=1.0)
    self.metrics["success"] = self.episode_success
    self.metrics["robot_speech_count"] = self.robot_speech_count
    self.metrics["silence_ratio"] = self.silence_count / steps
    self.metrics["repeated_speech_count"] = self.repeated_speech_count
    self.metrics["human_readiness"] = self.human_readiness
    self.metrics["human_reach_progress"] = self.reach_progress
    self.metrics["allostatic_load_total"] = self.allostatic_load_total
    self.metrics["attention_load"] = self.attention_load
    self.metrics["turn_taking_load"] = self.turn_taking_load
    self.metrics["proxemic_stress"] = self.proxemic_stress
    self.metrics["motor_adaptation_cost"] = self.motor_adaptation_cost
    self.metrics["human_waiting_cost"] = self.human_waiting_cost
    self.metrics["human_reach_effort"] = self.human_reach_effort
    self.metrics["robot_object_grasped"] = self.robot_object_grasped.float()

  def _reset_episode_state(self, env_ids: torch.Tensor) -> None:
    self._last_update_step = -1
    self.phase[env_ids] = int(HandoverPhase.APPROACH)
    self.phase_step[env_ids] = 0.0
    self.reach_progress[env_ids] = 0.0
    self.retreat_progress[env_ids] = 0.0
    self.episode_success[env_ids] = 0.0
    self.object_attached[env_ids] = False
    self.robot_object_grasped[env_ids] = False
    self.initial_object_z[env_ids] = 0.0
    self.max_object_lift[env_ids] = 0.0
    self.release_at_hand_event[env_ids] = 0.0
    initial_readiness = 1.0 if self.cfg.pure_task_mode else self.cfg.readiness_initial
    self.human_readiness[env_ids] = initial_readiness
    self.readiness_belief[env_ids] = initial_readiness
    self.readiness_hold[env_ids] = 0
    self.withdrawal_recovery[env_ids] = 0.0
    self.withdrawal_recovery_hold[env_ids] = 0
    self.human_state_id[env_ids] = (
      int(HumanState.READY) if self.cfg.pure_task_mode else int(HumanState.HESITANT)
    )
    self.last_speech_token[env_ids] = int(RobotSpeechToken.SILENCE)
    self.previous_speech_token[env_ids] = int(RobotSpeechToken.SILENCE)
    self.current_speech_is_silence[env_ids] = 1.0
    self.current_speech_is_repeated[env_ids] = 0.0
    self.current_speech_is_active[env_ids] = 0.0
    self.robot_speech_count[env_ids] = 0.0
    self.silence_count[env_ids] = 0.0
    self.repeated_speech_count[env_ids] = 0.0
    self.handoff_count[env_ids] = 0.0
    for tensor in (
      self.attention_load,
      self.turn_taking_load,
      self.proxemic_stress,
      self.motor_adaptation_cost,
      self.human_waiting_cost,
      self.human_reach_effort,
      self.allostatic_load_total,
    ):
      tensor[env_ids] = 0.0

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    self._reset_episode_state(env_ids)

    r = self.cfg.object_pose_range
    lower = torch.tensor([r.x[0], r.y[0], r.z[0]], device=self.device)
    upper = torch.tensor([r.x[1], r.y[1], r.z[1]], device=self.device)
    pos = sample_uniform(lower, upper, (n, 3), device=self.device)
    pos = pos + self._env.scene.env_origins[env_ids]
    yaw = sample_uniform(r.yaw[0], r.yaw[1], (n,), device=self.device)
    quat = quat_from_euler_xyz(
      torch.zeros(n, device=self.device),
      torch.zeros(n, device=self.device),
      yaw,
    )
    pose = torch.cat([pos, quat], dim=-1)
    velocity = torch.zeros(n, 6, device=self.device)
    self.object.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    self.object.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)
    self.initial_object_z[env_ids] = pos[:, 2]
    self._start_with_robot_grasped_object(env_ids)

    self.hand_pos[env_ids] = self._interpolate_hand_position()[env_ids]
    self.target_pos[env_ids] = self.hand_pos[env_ids]
    self._write_visual_state()
    self._update_episode_metrics()

  def _update_object_lift(self, object_pos: torch.Tensor) -> None:
    lift = torch.clamp(object_pos[:, 2] - self.initial_object_z, min=0.0)
    active = self.robot_object_grasped | self.object_attached
    self.max_object_lift = torch.where(
      active,
      torch.maximum(self.max_object_lift, lift),
      self.max_object_lift,
    )

  def _update_metrics(self) -> None:
    self.pre_reward_update()

  def _update_command(self) -> None:
    self.pre_reward_update()

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    for batch in env_indices:
      visualizer.add_sphere(
        center=self.hand_pos[batch].detach().cpu().numpy(),
        radius=0.035,
        color=(0.1, 0.8, 1.0, 0.45),
        label=f"handover_hand_target_{batch}",
      )


class HrgymFullHandoverCommand(AllostaticHandoverCommand):
  """HRGym RobotHumanHandover-style command with pkl-driven human animation."""

  cfg: HrgymFullHandoverCommandCfg

  def __init__(self, cfg: HrgymFullHandoverCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.human: Entity = env.scene[cfg.human_entity_name]
    self.animation_library = HrgymAnimationLibrary(
      vendor_root=cfg.animation_vendor_root,
      animation_names=cfg.animation_names,
      device=self.device,
    )

    if tuple(self.human.joint_names) != HRGYM_HUMAN_JOINT_NAMES:
      missing = sorted(set(HRGYM_HUMAN_JOINT_NAMES) - set(self.human.joint_names))
      extra = sorted(set(self.human.joint_names) - set(HRGYM_HUMAN_JOINT_NAMES))
      raise ValueError(
        "HRGym human joint names do not match the animation pkl schema. "
        f"missing={missing[:5]} extra={extra[:5]}"
      )

    left_ids, _ = self.human.find_bodies("L_Hand")
    right_ids, _ = self.human.find_bodies("R_Hand")
    left_site_ids, _ = self.human.find_sites("L_Hand")
    right_site_ids, _ = self.human.find_sites("R_Hand")
    self._left_hand_body_id = left_ids[0]
    self._right_hand_body_id = right_ids[0]
    self._left_hand_site_id = left_site_ids[0]
    self._right_hand_site_id = right_site_ids[0]
    self._yrot_left = quat_from_euler_xyz(
      torch.zeros(self.num_envs, device=self.device),
      torch.full((self.num_envs,), -torch.pi / 2.0, device=self.device),
      torch.zeros(self.num_envs, device=self.device),
    )
    self._yrot_right = quat_from_euler_xyz(
      torch.zeros(self.num_envs, device=self.device),
      torch.full((self.num_envs,), torch.pi / 2.0, device=self.device),
      torch.zeros(self.num_envs, device=self.device),
    )
    self._human_base_pos_offset = torch.tensor(
      cfg.human_base_pos_offset,
      dtype=torch.float32,
      device=self.device,
    )
    self._human_base_quat = torch.tensor(
      cfg.human_base_quat_wxyz,
      dtype=torch.float32,
      device=self.device,
    ).repeat(self.num_envs, 1)
    self._joint_zero_vel = torch.zeros(
      self.num_envs,
      len(HRGYM_HUMAN_JOINT_NAMES),
      device=self.device,
    )

    self.animation_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.animation_frame = torch.zeros(self.num_envs, device=self.device)
    self._classic_animation_frame = torch.zeros(self.num_envs, device=self.device)
    self._delayed_animation_frames = torch.zeros(self.num_envs, device=self.device)
    self.palm_distance = torch.full((self.num_envs,), 1.0, device=self.device)
    self.initial_palm_distance = torch.full((self.num_envs,), 1.0, device=self.device)
    self.best_palm_distance = torch.full((self.num_envs,), 1.0, device=self.device)
    self.robot_reached_hand = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self._frozen_hand_pos = torch.zeros(self.num_envs, 3, device=self.device)

    self.metrics["animation_current_id"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["animation_frame"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["object_attached"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["palm_distance"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["robot_object_grasped"] = torch.zeros(self.num_envs, device=self.device)

  def _update_phase_and_handoff(self) -> None:
    self._advance_animation_frames()
    self._write_human_animation_state()
    live_hand_pos = self._read_palm_target()
    self.hand_pos[:] = (
      self._frozen_hand_pos if self.cfg.freeze_hand_target_after_reset else live_hand_pos
    )
    self.target_pos[:] = self.hand_pos

    ready = self.human_readiness >= self.cfg.readiness_threshold
    key0, key1, lengths = self._current_keyframe_tensors()
    if self.cfg.pure_task_mode:
      self.phase = torch.where(
        self.phase == int(HandoverPhase.APPROACH),
        torch.full_like(self.phase, int(HandoverPhase.REACH_OUT)),
        self.phase,
      )
    phase = self.phase

    can_reach = (phase == int(HandoverPhase.APPROACH)) & (
      self._classic_animation_frame > key0
    )
    if self.cfg.require_readiness_for_reach:
      can_reach &= ready
    self.phase = torch.where(
      can_reach,
      torch.full_like(self.phase, int(HandoverPhase.REACH_OUT)),
      self.phase,
    )
    self.phase_step = torch.where(can_reach, torch.zeros_like(self.phase_step), self.phase_step)
    self.phase_step += 1.0

    reaching = self.phase == int(HandoverPhase.REACH_OUT)
    midpoint = key0 + 0.5 * (key1 - key0)
    should_loop = reaching & (self._classic_animation_frame > midpoint)
    modulated = self._looped_animation_frame(
      self._classic_animation_frame,
      midpoint,
    )
    self.animation_frame = torch.where(should_loop, modulated, self._classic_animation_frame)
    self._delayed_animation_frames = torch.where(
      should_loop,
      self._classic_animation_frame - self.animation_frame,
      self._delayed_animation_frames,
    )

    retreating = self.phase == int(HandoverPhase.RETREAT)
    self.animation_frame = torch.where(
      retreating,
      self._classic_animation_frame - self._delayed_animation_frames,
      self.animation_frame,
    )
    self.animation_frame = torch.minimum(self.animation_frame, lengths - 1.0)

    denom = torch.clamp(key1 - key0, min=1.0)
    self.reach_progress = torch.clamp((self.animation_frame - key0) / denom, 0.0, 1.0)
    self.reach_progress = torch.where(
      self.phase == int(HandoverPhase.APPROACH),
      torch.zeros_like(self.reach_progress),
      self.reach_progress,
    )
    retreat_denom = torch.clamp(lengths - 1.0 - key1, min=1.0)
    self.retreat_progress = torch.where(
      retreating | (self.phase == int(HandoverPhase.COMPLETE)),
      torch.clamp((self.animation_frame - key1) / retreat_denom, 0.0, 1.0),
      self.retreat_progress,
    )

    object_pos = self._sync_robot_grasped_object(self.object.data.root_link_pos_w)
    self._update_object_lift(object_pos)
    handoff_pos = self._handoff_reference_pos(object_pos)
    self.palm_distance = torch.norm(handoff_pos - self.hand_pos, dim=-1)
    can_release = self._gripper_release_signal()
    self.best_palm_distance = torch.where(
      self.robot_object_grasped,
      torch.minimum(self.best_palm_distance, self.palm_distance),
      self.best_palm_distance,
    )
    effective_palm_distance = torch.where(
      can_release,
      torch.minimum(self.palm_distance, self.best_palm_distance),
      self.palm_distance,
    )
    reached_now = effective_palm_distance <= self.cfg.success_threshold
    if self.cfg.pure_task_min_distance_improvement > 0.0:
      reached_now &= (
        effective_palm_distance
        <= self.initial_palm_distance - self.cfg.pure_task_min_distance_improvement
      )
    self.robot_reached_hand |= self.robot_object_grasped & reached_now
    self.release_at_hand_event = (
      self.robot_object_grasped.float()
      * can_release.float()
      * self.robot_reached_hand.float()
    )
    distance_gate = self.robot_reached_hand if self.cfg.pure_task_mode else reached_now
    handoff_now = (
      reaching
      & ready
      & (self.reach_progress >= self.cfg.handoff_reach_progress_threshold)
      & distance_gate
      & can_release
    )
    lift_gate = torch.ones_like(handoff_now)
    if self.cfg.min_lift_before_handoff > 0.0:
      lift_gate = self.max_object_lift >= self.cfg.min_lift_before_handoff
      handoff_now &= lift_gate
    grasp_gate = torch.ones_like(handoff_now)
    if self.cfg.robot_grasp_latch_enabled:
      grasp_gate = self.robot_object_grasped
      handoff_now &= grasp_gate
    self._set_handoff_gate_diagnostics(
      ready=ready,
      reaching=reaching,
      reach_progress_gate=self.reach_progress >= self.cfg.handoff_reach_progress_threshold,
      distance_gate=distance_gate,
      release_gate=can_release,
      lift_gate=lift_gate,
      grasp_gate=grasp_gate,
      handoff_now=handoff_now,
    )
    newly_attached = handoff_now & ~self.object_attached
    self.object_attached |= handoff_now
    self.handoff_count += newly_attached.float()
    release_away = can_release & ~handoff_now
    if not self.cfg.allow_release_away_from_hand:
      release_away &= torch.zeros_like(release_away)
    self.robot_object_grasped &= ~(self.object_attached | release_away)

    to_retreat = self.object_attached & (self.phase == int(HandoverPhase.REACH_OUT))
    self.phase = torch.where(
      to_retreat,
      torch.full_like(self.phase, int(HandoverPhase.RETREAT)),
      self.phase,
    )
    self.phase_step = torch.where(to_retreat, torch.zeros_like(self.phase_step), self.phase_step)

    if self.cfg.pure_task_mode:
      complete = self.object_attached
    else:
      complete = (self.phase == int(HandoverPhase.RETREAT)) & (
        self.animation_frame >= lengths - 1.0
      )
    self.phase = torch.where(
      complete,
      torch.full_like(self.phase, int(HandoverPhase.COMPLETE)),
      self.phase,
    )
    self.episode_success = torch.maximum(self.episode_success, complete.float())

    if self.object_attached.any():
      attached_ids = self.object_attached.nonzero().flatten()
      self._write_entity_pose(self.object, self.hand_pos[attached_ids], attached_ids)

  def _advance_animation_frames(self) -> None:
    running = self.phase != int(HandoverPhase.COMPLETE)
    if self.cfg.require_readiness_for_animation_start:
      ready = self.human_readiness >= self.cfg.readiness_threshold
      running &= ready | (self.phase != int(HandoverPhase.APPROACH))
    self._classic_animation_frame = torch.where(
      running,
      self._classic_animation_frame + self.cfg.human_animation_freq * self._env.step_dt,
      self._classic_animation_frame,
    )

  def _current_keyframe_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key0 = torch.zeros(self.num_envs, device=self.device)
    key1 = torch.zeros(self.num_envs, device=self.device)
    lengths = torch.zeros(self.num_envs, device=self.device)
    for idx, animation in enumerate(self.animation_library.animations):
      mask = self.animation_id == idx
      if not mask.any():
        continue
      first, second = animation.keyframes
      key0 = torch.where(mask, torch.full_like(key0, float(first)), key0)
      key1 = torch.where(mask, torch.full_like(key1, float(second)), key1)
      lengths = torch.where(mask, torch.full_like(lengths, float(animation.num_frames)), lengths)
    return key0, key1, lengths

  def _looped_animation_frame(
    self,
    classic_frame: torch.Tensor,
    midpoint: torch.Tensor,
  ) -> torch.Tensor:
    out = classic_frame.clone()
    loop_scale = self._readiness_loop_amplitude_scale()
    for idx, animation in enumerate(self.animation_library.animations):
      mask = self.animation_id == idx
      if not mask.any():
        continue
      info = animation.info
      amplitudes = info.get("loop_amplitudes", [1.0])
      speeds = info.get("loop_speeds", [1.0])
      total = torch.zeros_like(out)
      for amplitude, speed in zip(amplitudes, speeds):
        amp = torch.full_like(out, float(amplitude)) * loop_scale
        amp = torch.clamp(amp, min=1.0e-3)
        spd = torch.full_like(out, float(speed))
        total += amp * torch.sin((classic_frame - midpoint) / (amp / spd)) + midpoint
      looped = total - midpoint * (len(amplitudes) - 1)
      out = torch.where(mask, looped, out)
    return out

  def _readiness_loop_amplitude_scale(self) -> torch.Tensor:
    readiness = torch.clamp(self.human_readiness, 0.0, 1.0)
    scale = (
      self.cfg.hesitant_loop_amplitude_scale
      - readiness
      * (self.cfg.hesitant_loop_amplitude_scale - self.cfg.ready_loop_amplitude_scale)
    )
    return torch.clamp(scale, min=1.0e-3)

  def _write_human_animation_state(self) -> None:
    root_pos = torch.zeros(self.num_envs, 3, device=self.device)
    root_quat = self._identity_quat.clone()
    joint_pos = torch.zeros(
      self.num_envs,
      len(HRGYM_HUMAN_JOINT_NAMES),
      device=self.device,
    )
    env_origin = self._env.scene.env_origins
    frames = torch.floor(self.animation_frame).long()
    for idx, animation in enumerate(self.animation_library.animations):
      env_ids = (self.animation_id == idx).nonzero().flatten()
      if len(env_ids) == 0:
        continue
      frame_ids = torch.clamp(frames[env_ids], 0, animation.num_frames - 1)
      info = animation.info
      scale = float(info.get("scale", 1.0))
      offset = torch.tensor(
        info.get("position_offset", [0.0, 0.0, 0.0]),
        dtype=torch.float32,
        device=self.device,
      )
      offset_quat = xyzw_to_wxyz(
        torch.tensor(
          info.get("orientation_quat", [0.0, 0.0, 0.0, 1.0]),
          dtype=torch.float32,
          device=self.device,
        )
      ).repeat(len(env_ids), 1)
      base_quat = self._human_base_quat[env_ids]
      local_pos = animation.root_pos[frame_ids] * scale + offset
      rotated_pos = quat_apply(base_quat, quat_apply(offset_quat, local_pos))
      root_pos[env_ids] = env_origin[env_ids] + self._human_base_pos_offset + rotated_pos
      pelvis_quat = xyzw_to_wxyz(animation.root_quat_xyzw[frame_ids])
      root_quat[env_ids] = quat_mul(base_quat, quat_mul(offset_quat, pelvis_quat))
      joint_pos[env_ids] = animation.joint_pos[frame_ids]

    root_pose = torch.cat([root_pos, root_quat], dim=-1)
    self.human.write_mocap_pose_to_sim(root_pose, env_ids=self._env_ids_all)
    self.human.write_joint_state_to_sim(
      joint_pos,
      self._joint_zero_vel,
      env_ids=self._env_ids_all,
    )

  def _read_palm_target(self) -> torch.Tensor:
    left_pos = self.human.data.site_pos_w[:, self._left_hand_site_id, :]
    right_pos = self.human.data.site_pos_w[:, self._right_hand_site_id, :]
    left_quat = self.human.data.body_link_quat_w[:, self._left_hand_body_id, :]
    right_quat = self.human.data.body_link_quat_w[:, self._right_hand_body_id, :]
    left_rot = quat_mul(left_quat, self._yrot_left)
    right_rot = quat_mul(right_quat, self._yrot_right)
    left_local_offset = torch.tensor(
      [0.02, -0.03, -0.03],
      dtype=torch.float32,
      device=self.device,
    ).expand(self.num_envs, 3)
    right_local_offset = torch.tensor(
      [-0.02, -0.03, -0.03],
      dtype=torch.float32,
      device=self.device,
    ).expand(self.num_envs, 3)
    left_offset = quat_apply(
      left_rot,
      left_local_offset,
    )
    right_offset = quat_apply(
      right_rot,
      right_local_offset,
    )
    left_target = left_pos + left_offset
    right_target = right_pos + right_offset
    target = right_target.clone()
    for idx, animation in enumerate(self.animation_library.animations):
      if animation.object_holding_hand != "left":
        continue
      mask = (self.animation_id == idx).unsqueeze(-1)
      target = torch.where(mask, left_target, target)
    return target

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    self._reset_episode_state(env_ids)
    self.animation_id[env_ids] = torch.randint(
      0,
      len(self.animation_library),
      (n,),
      device=self.device,
    )
    self.animation_frame[env_ids] = 0.0
    self._classic_animation_frame[env_ids] = 0.0
    self._delayed_animation_frames[env_ids] = 0.0
    self.palm_distance[env_ids] = 1.0
    self.best_palm_distance[env_ids] = 1.0
    self.robot_reached_hand[env_ids] = False
    self._write_human_animation_state()
    self._env.sim.forward()
    self._env.scene.update(0.0)

    r = self.cfg.object_pose_range
    lower = torch.tensor([r.x[0], r.y[0], r.z[0]], device=self.device)
    upper = torch.tensor([r.x[1], r.y[1], r.z[1]], device=self.device)
    pos = sample_uniform(lower, upper, (n, 3), device=self.device)
    pos = pos + self._env.scene.env_origins[env_ids]
    yaw = sample_uniform(r.yaw[0], r.yaw[1], (n,), device=self.device)
    quat = quat_from_euler_xyz(
      torch.zeros(n, device=self.device),
      torch.full((n,), torch.pi, device=self.device),
      yaw,
    )
    pose = torch.cat([pos, quat], dim=-1)
    velocity = torch.zeros(n, 6, device=self.device)
    self.object.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    self.object.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)
    self.initial_object_z[env_ids] = pos[:, 2]
    self.release_at_hand_event[env_ids] = 0.0
    live_hand_pos = self._read_palm_target()[env_ids]
    self._frozen_hand_pos[env_ids] = live_hand_pos
    self.hand_pos[env_ids] = live_hand_pos
    self.target_pos[env_ids] = live_hand_pos
    self._start_with_robot_grasped_object(env_ids)
    object_pos = torch.where(
      self.robot_object_grasped[env_ids].unsqueeze(-1),
      self._robot_ee_pos()[env_ids] + self._robot_grasp_object_offset,
      pos,
    )
    self.initial_palm_distance[env_ids] = torch.norm(
      object_pos - self._frozen_hand_pos[env_ids],
      dim=-1,
    )
    self._update_episode_metrics()

  def _write_visual_state(self) -> None:
    self._write_human_animation_state()

  def _update_episode_metrics(self) -> None:
    super()._update_episode_metrics()
    self.metrics["animation_current_id"] = self.animation_id.float()
    self.metrics["animation_frame"] = self.animation_frame
    self.metrics["object_attached"] = self.object_attached.float()
    self.metrics["palm_distance"] = self.palm_distance
    self.metrics["robot_object_grasped"] = self.robot_object_grasped.float()
