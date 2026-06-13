"""Metrics terms for Mjlab allostatic handover."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from allostatic_handover.envs.human_hidden_state import HumanState

from .commands import AllostaticHandoverCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _command(env: ManagerBasedRlEnv, command_name: str) -> AllostaticHandoverCommand:
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, AllostaticHandoverCommand):
    raise TypeError(
      f"Command '{command_name}' must be AllostaticHandoverCommand, got {type(command)}"
    )
  command.pre_reward_update()
  return command


def success(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return _command(env, command_name).episode_success


def robot_speech_count(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return _command(env, command_name).robot_speech_count


def silence_ratio(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _command(env, command_name)
  steps = torch.clamp(env.episode_length_buf.float() + 1.0, min=1.0)
  return command.silence_count / steps


def repeated_speech_count(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return _command(env, command_name).repeated_speech_count


def human_readiness(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return _command(env, command_name).human_readiness


def human_reach_progress(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return _command(env, command_name).reach_progress


def human_state_ratio(
  env: ManagerBasedRlEnv,
  command_name: str,
  state: HumanState | int,
) -> torch.Tensor:
  command = _command(env, command_name)
  return (command.human_state_id == int(state)).float()


def allostatic_load_total(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return _command(env, command_name).allostatic_load_total


def attention_load(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return _command(env, command_name).attention_load


def turn_taking_load(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return _command(env, command_name).turn_taking_load


def proxemic_stress(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return _command(env, command_name).proxemic_stress


def motor_adaptation_cost(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return _command(env, command_name).motor_adaptation_cost


def human_waiting_cost(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return _command(env, command_name).human_waiting_cost


def human_reach_effort(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return _command(env, command_name).human_reach_effort


def animation_current_id(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _command(env, command_name)
  return getattr(command, "animation_id", torch.zeros(env.num_envs, device=env.device)).float()


def animation_frame(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _command(env, command_name)
  return getattr(command, "animation_frame", torch.zeros(env.num_envs, device=env.device))


def object_attached(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _command(env, command_name)
  return command.object_attached.float()


def robot_object_grasped(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _command(env, command_name)
  return getattr(
    command,
    "robot_object_grasped",
    torch.zeros(env.num_envs, device=env.device),
  ).float()


def palm_distance(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _command(env, command_name)
  return getattr(command, "palm_distance", torch.zeros(env.num_envs, device=env.device))
