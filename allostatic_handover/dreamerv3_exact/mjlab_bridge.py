"""Bridge between Mjlab allostatic handover and exact DreamerV3.

This module intentionally keeps Mjlab and DreamerV3 as external dependencies.
Imports that require those projects are lazy so unit tests can exercise the
observation/replay contracts without importing the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


OBS_KEYS = ("public_obs", "reward", "is_first", "is_last", "is_terminal")
LABEL_KEYS = ("human_state_id", "human_readiness", "allostatic_load_total")
ACTION_KEY = "action"


@dataclass(kw_only=True)
class MjlabDreamerBridgeConfig:
  task_id: str = "Mjlab-Allostatic-Handover-Full-DreamerV3Allostatic"
  num_envs: int = 16
  seed: int = 101
  device: str = "cpu"
  render_mode: str | None = None
  object_name: str = "manipulation_object"
  command_name: str = "handover"
  project_root: str = "/mnt/k_iwamoto/sim_data/Projects/allostatic-handover"


def make_policy_obs(step: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
  """Return only the observation fields that the Dreamer policy may see."""
  return {key: step[key] for key in OBS_KEYS}


def make_replay_transition(
  step: dict[str, np.ndarray],
  action: np.ndarray,
) -> dict[str, np.ndarray]:
  """Combine public obs, action, and hidden labels for DreamerV3 replay.

  Hidden human-state values are not part of ``make_policy_obs()``. They are
  added here only as auxiliary labels consumed by ``AllostaticDreamerAgent``.
  """
  transition = {key: np.asarray(step[key]) for key in (*OBS_KEYS, *LABEL_KEYS)}
  transition[ACTION_KEY] = np.asarray(action, dtype=np.float32)
  return transition


def make_dreamer_spaces(public_obs_dim: int, action_dim: int = 5):
  """Create exact DreamerV3 spaces for public observations and 5D actions."""
  import elements

  obs_space = {
    "public_obs": elements.Space(np.float32, (int(public_obs_dim),)),
    "reward": elements.Space(np.float32, ()),
    "is_first": elements.Space(bool, (), 0, 2),
    "is_last": elements.Space(bool, (), 0, 2),
    "is_terminal": elements.Space(bool, (), 0, 2),
  }
  act_space = {
    "action": elements.Space(np.float32, (int(action_dim),), -1.0, 1.0),
  }
  return obs_space, act_space


def make_replay_stream(
  replay: Any,
  *,
  batch_size: int,
  batch_length: int,
  replay_context: int = 0,
  mode: str = "train",
):
  """Return a DreamerV3 stream with the required ``consec`` annotation."""
  import embodied

  source = embodied.streams.Stateless(lambda: replay.sample(batch_size, mode))
  return embodied.streams.Consec(
    source,
    length=batch_length,
    consec=1,
    prefix=replay_context,
    strict=True,
    contiguous=True,
  )


class MjlabDreamerBridge:
  """Vectorized online Mjlab environment wrapper for DreamerV3."""

  def __init__(self, config: MjlabDreamerBridgeConfig):
    self.config = config
    _set_mjlab_env_defaults()

    import torch
    import allostatic_handover.mjlab_tasks  # noqa: F401
    from allostatic_handover.mjlab_tasks import mdp
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    cfg = load_env_cfg(config.task_id, play=False)
    cfg.scene.num_envs = config.num_envs
    cfg.seed = config.seed
    self.env = ManagerBasedRlEnv(cfg=cfg, device=config.device, render_mode=config.render_mode)
    self.mdp = mdp
    self.torch = torch
    self.command = None
    self.num_envs = int(config.num_envs)
    self.action_dim = int(self.env.single_action_space.shape[0])
    if self.action_dim != 5:
      raise ValueError(
        f"DreamerV3 bridge requires 5D action, got {self.action_dim} for {config.task_id}"
      )
    self.last_extras_log: dict[str, Any] = {}
    self.last_done = np.zeros(self.num_envs, dtype=bool)
    self.last_terminal = np.zeros(self.num_envs, dtype=bool)
    self.current_step = self.reset()
    self.public_obs_dim = int(self.current_step["public_obs"].shape[-1])

  def reset(self) -> dict[str, np.ndarray]:
    self.env.reset(seed=self.config.seed)
    self.command = self.env.command_manager.get_term(self.config.command_name)
    zeros = self.torch.zeros(self.num_envs, dtype=self.torch.float32, device=self.env.device)
    flags = self.torch.zeros(self.num_envs, dtype=self.torch.bool, device=self.env.device)
    self.last_done[:] = False
    self.last_terminal[:] = False
    self.last_extras_log = dict(self.env.extras.get("log", {}))
    return self._observe(
      reward=zeros,
      is_first=self.torch.ones_like(flags),
      is_last=flags,
      is_terminal=flags,
    )

  def step(self, action: np.ndarray) -> dict[str, np.ndarray]:
    action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    if action.shape != (self.num_envs, self.action_dim):
      raise ValueError(
        f"expected action shape {(self.num_envs, self.action_dim)}, got {action.shape}"
      )
    action_tensor = self.torch.as_tensor(
      action,
      dtype=self.torch.float32,
      device=self.env.device,
    )
    _obs, reward, terminated, truncated, extras = self.env.step(action_tensor)
    done = terminated | truncated
    self.last_done = _to_numpy_bool(done)
    self.last_terminal = _to_numpy_bool(terminated)
    self.last_extras_log = dict(extras.get("log", {}))
    return self._observe(
      reward=reward,
      is_first=done,
      is_last=done,
      is_terminal=terminated,
    )

  def transition(
    self,
    step: dict[str, np.ndarray],
    action: dict[str, np.ndarray] | np.ndarray,
  ) -> dict[str, np.ndarray]:
    if isinstance(action, dict):
      action_array = action[ACTION_KEY]
    else:
      action_array = action
    return make_replay_transition(step, action_array)

  def random_action(self) -> dict[str, np.ndarray]:
    return {
      ACTION_KEY: np.random.uniform(
        -1.0,
        1.0,
        size=(self.num_envs, self.action_dim),
      ).astype(np.float32)
    }

  def current_metrics(self) -> dict[str, float]:
    command = self._command()
    command.pre_reward_update()
    result = {
      "env/success": _mean(command.episode_success),
      "env/robot_speech_count": _mean(command.robot_speech_count),
      "env/repeated_speech_count": _mean(command.repeated_speech_count),
      "env/silence_ratio": _mean(command.silence_count / self._episode_steps()),
      "env/load_mean": _mean(command.allostatic_load_total),
      "env/human_readiness": _mean(command.human_readiness),
      "env/reach_progress": _mean(command.reach_progress),
      "env/palm_distance": _mean(getattr(command, "palm_distance")),
      "env/object_attached": _mean(command.object_attached.float()),
      "env/robot_object_grasped": _mean(command.robot_object_grasped.float()),
    }
    human_state = command.human_state_id.detach().cpu().numpy()
    for state_id in range(6):
      result[f"env/human_state/{state_id}_ratio"] = float((human_state == state_id).mean())
    result.update(_flatten_log_scalars(self.last_extras_log, prefix="episode"))
    return result

  def close(self) -> None:
    self.env.close()

  def _observe(
    self,
    *,
    reward: Any,
    is_first: Any,
    is_last: Any,
    is_terminal: Any,
  ) -> dict[str, np.ndarray]:
    command = self._command()
    command.pre_reward_update()
    public_obs = self.mdp.world_model_public_obs(
      self.env,
      object_name=self.config.object_name,
      command_name=self.config.command_name,
    )
    return {
      "public_obs": _to_numpy_float(public_obs),
      "reward": _to_numpy_float(reward),
      "is_first": _to_numpy_bool(is_first),
      "is_last": _to_numpy_bool(is_last),
      "is_terminal": _to_numpy_bool(is_terminal),
      "human_state_id": _to_numpy_int(command.human_state_id),
      "human_readiness": _to_numpy_float(command.human_readiness),
      "allostatic_load_total": _to_numpy_float(command.allostatic_load_total),
    }

  def _command(self):
    if self.command is None:
      self.command = self.env.command_manager.get_term(self.config.command_name)
    return self.command

  def _episode_steps(self):
    return self.torch.clamp(self.env.episode_length_buf.float() + 1.0, min=1.0)


def _set_mjlab_env_defaults() -> None:
  import os

  os.environ.setdefault("MUJOCO_GL", "egl")
  os.environ.setdefault("XDG_CACHE_HOME", "/mnt/k_iwamoto/sim_data/tmp/xdg_cache")
  os.environ.setdefault("MPLCONFIGDIR", "/mnt/k_iwamoto/sim_data/tmp/matplotlib")


def _to_numpy_float(tensor: Any) -> np.ndarray:
  if hasattr(tensor, "detach"):
    return tensor.detach().cpu().numpy().astype(np.float32, copy=False)
  return np.asarray(tensor, dtype=np.float32)


def _to_numpy_bool(tensor: Any) -> np.ndarray:
  if hasattr(tensor, "detach"):
    return tensor.detach().cpu().numpy().astype(bool, copy=False)
  return np.asarray(tensor, dtype=bool)


def _to_numpy_int(tensor: Any) -> np.ndarray:
  if hasattr(tensor, "detach"):
    return tensor.detach().cpu().numpy().astype(np.int32, copy=False)
  return np.asarray(tensor, dtype=np.int32)


def _mean(tensor: Any) -> float:
  if hasattr(tensor, "detach"):
    return float(tensor.detach().float().mean().cpu().item())
  return float(np.asarray(tensor, dtype=np.float32).mean())


def _flatten_log_scalars(log: dict[str, Any], *, prefix: str) -> dict[str, float]:
  result: dict[str, float] = {}
  for key, value in log.items():
    try:
      array = np.asarray(value.detach().cpu() if hasattr(value, "detach") else value)
    except Exception:
      continue
    if array.size == 0:
      continue
    if np.issubdtype(array.dtype, np.number) or array.dtype == bool:
      result[f"{prefix}/{key}"] = float(array.astype(np.float32).mean())
  return result


def checkpoint_payload(
  *,
  agent: Any,
  config: dict[str, Any],
  public_obs_dim: int,
  action_dim: int,
  replay_dir: str | Path | None,
  metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
  return {
    "format": "allostatic_handover_exact_dreamerv3_online_policy_v1",
    "agent": agent.save(),
    "config": config,
    "obs_space": {"public_obs_dim": int(public_obs_dim)},
    "act_space": {"action_dim": int(action_dim)},
    "replay_dir": str(replay_dir) if replay_dir is not None else None,
    "metrics": metrics or {},
  }
