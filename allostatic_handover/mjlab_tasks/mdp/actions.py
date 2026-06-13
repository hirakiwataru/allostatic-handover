"""Mjlab action terms for allostatic handover tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import DifferentialIKAction, DifferentialIKActionCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class TabletopDifferentialIKActionCfg(DifferentialIKActionCfg):
  """Differential IK action with a world-frame z floor for tabletop tasks."""

  min_frame_z: float = 0.0
  """Minimum desired world z position for the controlled frame."""

  def build(self, env: ManagerBasedRlEnv) -> TabletopDifferentialIKAction:
    return TabletopDifferentialIKAction(self, env)


class TabletopDifferentialIKAction(DifferentialIKAction):
  """Clamps the IK target above the table before solving joint targets."""

  cfg: TabletopDifferentialIKActionCfg

  def process_actions(self, actions: torch.Tensor) -> None:
    super().process_actions(actions)
    self._desired_pos[:, 2].clamp_(min=self.cfg.min_frame_z)


@dataclass(kw_only=True)
class SpeechActionCfg(ActionTermCfg):
  """One-dimensional continuous action that is later binned into speech tokens."""

  def build(self, env: ManagerBasedRlEnv) -> SpeechAction:
    return SpeechAction(self, env)


class SpeechAction(ActionTerm):
  """Stores a raw scalar speech action without writing anything to MuJoCo."""

  def __init__(self, cfg: SpeechActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._raw_actions = torch.full((self.num_envs, 1), -1.0, device=self.device)

  @property
  def action_dim(self) -> int:
    return 1

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  @property
  def speech_scalar(self) -> torch.Tensor:
    return self._raw_actions[:, 0]

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = torch.clamp(actions, -1.0, 1.0)

  def apply_actions(self) -> None:
    pass

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_actions[env_ids] = -1.0
