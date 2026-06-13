"""Termination terms for Mjlab allostatic handover."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .commands import AllostaticHandoverCommand, HandoverPhase

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def handover_complete(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, AllostaticHandoverCommand):
    raise TypeError(
      f"Command '{command_name}' must be AllostaticHandoverCommand, got {type(command)}"
    )
  command.pre_reward_update()
  return command.phase == int(HandoverPhase.COMPLETE)
