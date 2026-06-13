"""Reward terms for Mjlab allostatic handover."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .commands import AllostaticHandoverCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_EE_CFG = SceneEntityCfg("robot", site_names=("grasp_site",))


def _command(env: ManagerBasedRlEnv, command_name: str) -> AllostaticHandoverCommand:
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, AllostaticHandoverCommand):
    raise TypeError(
      f"Command '{command_name}' must be AllostaticHandoverCommand, got {type(command)}"
    )
  command.pre_reward_update()
  return command


def staged_handover_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  reaching_std: float,
  handoff_std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_EE_CFG,
) -> torch.Tensor:
  """Dense reward for reaching the cube and bringing it to the human hand."""
  command = _command(env, command_name)
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  ee_pos = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  obj_pos = obj.data.root_link_pos_w
  reach_error = torch.sum(torch.square(ee_pos - obj_pos), dim=-1)
  handoff_error = torch.sum(torch.square(command.hand_pos - obj_pos), dim=-1)
  reaching = torch.exp(-reach_error / reaching_std**2)
  handoff = torch.exp(-handoff_error / handoff_std**2)
  readiness_gate = 0.35 + 0.65 * command.human_readiness
  return reaching * (0.7 + 1.3 * readiness_gate * handoff) + 0.25 * command.reach_progress


def robot_grasp_approach_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_EE_CFG,
) -> torch.Tensor:
  command = _command(env, command_name)
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  ee_pos = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  object_pos = obj.data.root_link_pos_w
  distance_error = torch.sum(torch.square(ee_pos - object_pos), dim=-1)
  close_bonus = 0.5 * command._gripper_close_signal().float()
  needs_grasp = (~command.robot_object_grasped & ~command.object_attached).float()
  return needs_grasp * torch.exp(-distance_error / std**2) * (1.0 + close_bonus)


def robot_carry_to_hand_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  command = _command(env, command_name)
  palm_distance = getattr(
    command,
    "palm_distance",
    torch.norm(command.hand_pos - command.object.data.root_link_pos_w, dim=-1),
  )
  carrying = command.robot_object_grasped.float()
  target_radius = max(float(command.cfg.success_threshold), 1e-6)
  normalized_margin = (target_radius - palm_distance) / target_radius
  shaped_proximity = torch.exp(-(palm_distance**2) / std**2)
  if command.cfg.pure_task_mode:
    # This is only a shaping term for approaching the receiving hand. Once the
    # robot has reached the hand, continuing to hold the object must not be more
    # profitable than releasing and completing the episode.
    reached = getattr(
      command,
      "robot_reached_hand",
      torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
    )
    return carrying * (~reached).float() * torch.clamp(normalized_margin, min=0.0)
  return carrying * (normalized_margin + 0.25 * shaped_proximity)


def release_at_hand_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  command = _command(env, command_name)
  palm_distance = getattr(
    command,
    "palm_distance",
    torch.norm(command.hand_pos - command.object.data.root_link_pos_w, dim=-1),
  )
  best_distance = getattr(command, "best_palm_distance", palm_distance)
  effective_distance = torch.minimum(palm_distance, best_distance)
  near_hand = effective_distance <= command.cfg.success_threshold
  if command.cfg.pure_task_min_distance_improvement > 0.0:
    initial_distance = getattr(command, "initial_palm_distance", palm_distance)
    near_hand &= (
      effective_distance
      <= initial_distance - command.cfg.pure_task_min_distance_improvement
    )
  if command.cfg.pure_task_mode and hasattr(command, "robot_reached_hand"):
    near_hand |= command.robot_reached_hand
  near_hand = near_hand.float()
  return command.release_at_hand_event * near_hand * torch.exp(
    -(effective_distance**2) / std**2
  )


def release_intent_at_hand_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Dense gripper-opening reward once the carried object has reached the hand."""
  command = _command(env, command_name)
  if command.cfg.gripper_action_name not in env.action_manager.active_terms:
    return torch.zeros(env.num_envs, device=env.device)

  gripper = env.action_manager.get_term(command.cfg.gripper_action_name)
  raw_action = gripper.raw_action[:, 0]
  threshold = float(command.cfg.release_action_threshold)
  denom = max(1.0 - threshold, 1e-6)
  release_intent = torch.clamp((raw_action - threshold) / denom, min=0.0, max=1.0)
  reached = getattr(
    command,
    "robot_reached_hand",
    command.palm_distance <= command.cfg.success_threshold,
  )
  return command.robot_object_grasped.float() * reached.float() * release_intent


def handover_precision_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  std: float,
) -> torch.Tensor:
  command = _command(env, command_name)
  obj: Entity = env.scene[object_name]
  error = torch.sum(torch.square(command.hand_pos - obj.data.root_link_pos_w), dim=-1)
  return torch.exp(-error / std**2)


def success_bonus(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _command(env, command_name)
  return command.episode_success


def robot_grasp_bonus(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _command(env, command_name)
  return getattr(
    command,
    "robot_object_grasped",
    torch.zeros(env.num_envs, device=env.device),
  ).float()


def handoff_bonus(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _command(env, command_name)
  return command.object_attached.float()


def speech_penalty(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _command(env, command_name)
  if command.cfg.reward_variant in {"speech_penalty", "allostatic"}:
    speech_load = command.attention_load + command.turn_taking_load
    excess = torch.clamp(
      speech_load - command.cfg.speech_penalty_load_threshold,
      min=0.0,
      max=command.cfg.speech_penalty_max_excess,
    )
    return torch.expm1(excess * command.cfg.speech_penalty_exp_scale)
  return torch.zeros(env.num_envs, device=env.device)


def allostatic_load_penalty(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _command(env, command_name)
  if command.cfg.reward_variant != "allostatic":
    return torch.zeros(env.num_envs, device=env.device)
  return command.allostatic_load_total


def waiting_cost(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _command(env, command_name)
  if command.cfg.reward_variant == "task_only":
    return torch.zeros(env.num_envs, device=env.device)
  return command.human_waiting_cost


def time_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.ones(env.num_envs, device=env.device)
