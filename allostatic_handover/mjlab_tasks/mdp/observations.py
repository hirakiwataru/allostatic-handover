"""Observation terms for Mjlab allostatic handover."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.utils.lab_api.math import quat_apply, quat_inv

from allostatic_handover.world_model.belief_model import (
  DEFAULT_BELIEF_MODEL_PATH,
  BeliefEstimator,
  load_belief_model,
)

from .commands import AllostaticHandoverCommand, HandoverPhase

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_EE_CFG = SceneEntityCfg("robot", site_names=("grasp_site",))
_DEFAULT_ROBOT_JOINTS_CFG = SceneEntityCfg("robot", joint_names=(".*",))
_BELIEF_MODEL_CACHE: dict[tuple[str, str], tuple[BeliefEstimator, dict[str, Any]]] = {}


def _command(env: ManagerBasedRlEnv, command_name: str) -> AllostaticHandoverCommand:
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, AllostaticHandoverCommand):
    raise TypeError(
      f"Command '{command_name}' must be AllostaticHandoverCommand, got {type(command)}"
    )
  command.pre_reward_update()
  return command


def ee_to_hand(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_EE_CFG,
) -> torch.Tensor:
  command = _command(env, command_name)
  robot: Entity = env.scene[asset_cfg.name]
  ee_pos_w = _single_site_pos_w(robot, asset_cfg)
  base_quat_w = robot.data.root_link_quat_w
  return quat_apply(quat_inv(base_quat_w), command.hand_pos - ee_pos_w)


def object_to_hand(
  env: ManagerBasedRlEnv,
  object_name: str,
  command_name: str,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  command = _command(env, command_name)
  obj: Entity = env.scene[object_name]
  robot: Entity = env.scene[asset_cfg.name]
  base_quat_w = robot.data.root_link_quat_w
  return quat_apply(quat_inv(base_quat_w), command.hand_pos - obj.data.root_link_pos_w)


def readiness_belief(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _command(env, command_name)
  return command.readiness_belief.unsqueeze(-1)


def load_proxy(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _command(env, command_name)
  proxy = torch.stack(
    [
      command.attention_load,
      command.turn_taking_load,
      command.proxemic_stress,
      command.human_waiting_cost,
      command.human_reach_effort,
    ],
    dim=-1,
  )
  return torch.clamp(proxy / 5.0, 0.0, 1.5)


def speech_context(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _command(env, command_name)
  max_token = 6.0
  return torch.stack(
    [
      command.last_speech_token.float() / max_token,
      command.previous_speech_token.float() / max_token,
      command.current_speech_is_repeated,
    ],
    dim=-1,
  )


def phase_progress(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _command(env, command_name)
  phase = command.phase.float() / float(HandoverPhase.COMPLETE)
  return torch.stack([phase, command.reach_progress, command.retreat_progress], dim=-1)


def world_model_public_obs(
  env: ManagerBasedRlEnv,
  object_name: str,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_EE_CFG,
  robot_joints_cfg: SceneEntityCfg = _DEFAULT_ROBOT_JOINTS_CFG,
) -> torch.Tensor:
  """Return the public observation used by the hidden-state world model.

  This intentionally excludes true human FSM state, true readiness, allostatic
  load, and hand-coded readiness/load proxies. The matching rollout dataset
  stores those hidden values only as training labels for the world model.
  """
  return torch.cat(
    [
      velocity_mdp.joint_pos_rel(env, asset_cfg=robot_joints_cfg),
      velocity_mdp.joint_vel_rel(env, asset_cfg=robot_joints_cfg),
      _ee_to_object(env, object_name=object_name, asset_cfg=asset_cfg),
      ee_to_hand(env, command_name=command_name, asset_cfg=asset_cfg),
      object_to_hand(env, object_name=object_name, command_name=command_name),
      speech_context(env, command_name=command_name),
      phase_progress(env, command_name=command_name),
    ],
    dim=-1,
  )


def wm_belief(
  env: ManagerBasedRlEnv,
  object_name: str,
  command_name: str,
  model_path: str = DEFAULT_BELIEF_MODEL_PATH,
  asset_cfg: SceneEntityCfg = _DEFAULT_EE_CFG,
  robot_joints_cfg: SceneEntityCfg = _DEFAULT_ROBOT_JOINTS_CFG,
  belief_dim: int = 16,
  num_human_states: int = 6,
) -> torch.Tensor:
  output = _world_model_output(
    env,
    object_name=object_name,
    command_name=command_name,
    model_path=model_path,
    asset_cfg=asset_cfg,
    robot_joints_cfg=robot_joints_cfg,
    belief_dim=belief_dim,
    num_human_states=num_human_states,
  )
  return output["belief"]


def wm_human_state_probs(
  env: ManagerBasedRlEnv,
  object_name: str,
  command_name: str,
  model_path: str = DEFAULT_BELIEF_MODEL_PATH,
  asset_cfg: SceneEntityCfg = _DEFAULT_EE_CFG,
  robot_joints_cfg: SceneEntityCfg = _DEFAULT_ROBOT_JOINTS_CFG,
  belief_dim: int = 16,
  num_human_states: int = 6,
) -> torch.Tensor:
  output = _world_model_output(
    env,
    object_name=object_name,
    command_name=command_name,
    model_path=model_path,
    asset_cfg=asset_cfg,
    robot_joints_cfg=robot_joints_cfg,
    belief_dim=belief_dim,
    num_human_states=num_human_states,
  )
  return output["human_state_probs"]


def wm_readiness_pred(
  env: ManagerBasedRlEnv,
  object_name: str,
  command_name: str,
  model_path: str = DEFAULT_BELIEF_MODEL_PATH,
  asset_cfg: SceneEntityCfg = _DEFAULT_EE_CFG,
  robot_joints_cfg: SceneEntityCfg = _DEFAULT_ROBOT_JOINTS_CFG,
  belief_dim: int = 16,
  num_human_states: int = 6,
) -> torch.Tensor:
  output = _world_model_output(
    env,
    object_name=object_name,
    command_name=command_name,
    model_path=model_path,
    asset_cfg=asset_cfg,
    robot_joints_cfg=robot_joints_cfg,
    belief_dim=belief_dim,
    num_human_states=num_human_states,
  )
  return output["readiness"]


def wm_load_pred(
  env: ManagerBasedRlEnv,
  object_name: str,
  command_name: str,
  model_path: str = DEFAULT_BELIEF_MODEL_PATH,
  asset_cfg: SceneEntityCfg = _DEFAULT_EE_CFG,
  robot_joints_cfg: SceneEntityCfg = _DEFAULT_ROBOT_JOINTS_CFG,
  belief_dim: int = 16,
  num_human_states: int = 6,
) -> torch.Tensor:
  output = _world_model_output(
    env,
    object_name=object_name,
    command_name=command_name,
    model_path=model_path,
    asset_cfg=asset_cfg,
    robot_joints_cfg=robot_joints_cfg,
    belief_dim=belief_dim,
    num_human_states=num_human_states,
  )
  return output["load"]


def privileged_human_state(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _command(env, command_name)
  return command.human_state_id.float().unsqueeze(-1) / 5.0


def privileged_load(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _command(env, command_name)
  return command.allostatic_load_total.unsqueeze(-1)


def _world_model_output(
  env: ManagerBasedRlEnv,
  *,
  object_name: str,
  command_name: str,
  model_path: str,
  asset_cfg: SceneEntityCfg,
  robot_joints_cfg: SceneEntityCfg,
  belief_dim: int,
  num_human_states: int,
) -> dict[str, torch.Tensor]:
  step = int(getattr(env, "common_step_counter", -1))
  cache_key = (
    step,
    object_name,
    command_name,
    _resolve_model_path(model_path),
  )
  if getattr(env, "_allostatic_wm_output_key", None) == cache_key:
    return env._allostatic_wm_output

  public_obs = world_model_public_obs(
    env,
    object_name=object_name,
    command_name=command_name,
    asset_cfg=asset_cfg,
    robot_joints_cfg=robot_joints_cfg,
  )
  resolved_path = cache_key[-1]
  model = _load_world_model_or_none(resolved_path, public_obs.device)
  if model is None:
    output = _zero_world_model_output(
      public_obs.shape[0],
      public_obs.device,
      public_obs.dtype,
      belief_dim=belief_dim,
      num_human_states=num_human_states,
    )
  else:
    action = velocity_mdp.last_action(env)
    if public_obs.shape[-1] != model.config.public_obs_dim:
      raise ValueError(
        f"World-model public_obs dim mismatch: env produced {public_obs.shape[-1]}, "
        f"model expects {model.config.public_obs_dim}"
      )
    if action.shape[-1] != model.config.action_dim:
      raise ValueError(
        f"World-model action dim mismatch: env produced {action.shape[-1]}, "
        f"model expects {model.config.action_dim}"
      )
    hidden = getattr(env, "_allostatic_wm_hidden", None)
    if hidden is None or hidden.shape[0] != public_obs.shape[0]:
      hidden = model.initial_state(public_obs.shape[0], public_obs.device)
    reset_mask = None
    if hasattr(env, "episode_length_buf"):
      reset_mask = env.episode_length_buf == 0
    with torch.inference_mode():
      output, hidden = model.step(public_obs, action, hidden, reset_mask=reset_mask)
    env._allostatic_wm_hidden = hidden.detach()

  env._allostatic_wm_output_key = cache_key
  env._allostatic_wm_output = output
  return output


def _resolve_model_path(model_path: str) -> str:
  env_path = os.environ.get("ALLOSTATIC_WM_BELIEF_MODEL")
  return env_path or model_path


def _load_world_model_or_none(
  model_path: str,
  device: torch.device,
) -> BeliefEstimator | None:
  if not model_path:
    return None
  path = Path(model_path)
  if not path.exists():
    return None
  key = (str(path.resolve()), str(device))
  cached = _BELIEF_MODEL_CACHE.get(key)
  if cached is None:
    model, metadata = load_belief_model(path, map_location=device)
    model.to(device)
    model.eval()
    _BELIEF_MODEL_CACHE[key] = (model, metadata)
    return model
  return cached[0]


def _zero_world_model_output(
  num_envs: int,
  device: torch.device,
  dtype: torch.dtype,
  *,
  belief_dim: int,
  num_human_states: int,
) -> dict[str, torch.Tensor]:
  probs = torch.zeros(num_envs, num_human_states, device=device, dtype=dtype)
  if num_human_states > 0:
    probs[:, 0] = 1.0
  return {
    "belief": torch.zeros(num_envs, belief_dim, device=device, dtype=dtype),
    "human_state_probs": probs,
    "readiness": torch.zeros(num_envs, 1, device=device, dtype=dtype),
    "load": torch.zeros(num_envs, 1, device=device, dtype=dtype),
  }


def _ee_to_object(
  env: ManagerBasedRlEnv,
  object_name: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  ee_pos_w = _single_site_pos_w(robot, asset_cfg)
  distance_vec_w = obj.data.root_link_pos_w - ee_pos_w
  base_quat_w = robot.data.root_link_quat_w
  return quat_apply(quat_inv(base_quat_w), distance_vec_w)


def _single_site_pos_w(robot: Entity, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  site_ids = asset_cfg.site_ids
  if isinstance(site_ids, slice):
    if asset_cfg.site_names is None:
      raise ValueError("asset_cfg must specify exactly one site name or site id")
    site_names = (
      (asset_cfg.site_names,)
      if isinstance(asset_cfg.site_names, str)
      else tuple(asset_cfg.site_names)
    )
    if len(site_names) != 1:
      raise ValueError(f"expected one site name, got {site_names}")
    site_id = robot.site_names.index(site_names[0])
  elif len(site_ids) == 1:
    site_id = int(site_ids[0])
  else:
    raise ValueError(f"expected one site id, got {site_ids}")
  return robot.data.site_pos_w[:, site_id]
