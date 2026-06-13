#!/usr/bin/env python3
"""Collect Mjlab rollouts for allostatic world-model training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--task-id",
    default="Mjlab-Allostatic-Handover-Full-TaskOnlySpeech",
  )
  parser.add_argument(
    "--output",
    type=Path,
    default=Path("outputs/world_model/smoke/task_only_speech_dataset.npz"),
  )
  parser.add_argument("--num-envs", type=int, default=8)
  parser.add_argument("--steps", type=int, default=1024)
  parser.add_argument("--seed", type=int, default=101)
  parser.add_argument("--device", default=None)
  parser.add_argument(
    "--policy",
    choices=("mixed", "scripted", "random", "excessive_speech"),
    default="mixed",
  )
  parser.add_argument("--delta-pos-scale", type=float, default=0.10)
  parser.add_argument("--release-distance", type=float, default=0.28)
  parser.add_argument("--wandb-mode", choices=("disabled", "offline", "online"), default="disabled")
  parser.add_argument("--wandb-project", default="allostatic-handover-mjlab")
  parser.add_argument("--wandb-group", default=None)
  parser.add_argument("--wandb-run-name", default=None)
  args = parser.parse_args()

  os.environ.setdefault("MUJOCO_GL", "egl")
  os.environ.setdefault("XDG_CACHE_HOME", "/mnt/k_iwamoto/sim_data/tmp/xdg_cache")
  os.environ.setdefault("MPLCONFIGDIR", "/mnt/k_iwamoto/sim_data/tmp/matplotlib")

  import allostatic_handover.mjlab_tasks  # noqa: F401
  from allostatic_handover.envs.speech_events import (
    RobotSpeechToken,
    robot_speech_to_scalar,
  )
  from allostatic_handover.mjlab_tasks import mdp
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg
  from mjlab.utils.torch import configure_torch_backends

  configure_torch_backends()
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  cfg = load_env_cfg(args.task_id, play=False)
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed

  env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
  arrays: dict[str, list[np.ndarray]] = {
    "public_obs": [],
    "action": [],
    "reward": [],
    "done": [],
    "human_state_id": [],
    "human_readiness": [],
    "allostatic_load_total": [],
    "phase": [],
    "reach_progress": [],
  }
  try:
    env.reset()
    command = env.command_manager.get_term("handover")
    silence = robot_speech_to_scalar(RobotSpeechToken.SILENCE)
    announce = robot_speech_to_scalar(RobotSpeechToken.ANNOUNCE_HANDOVER)
    ask_ready = robot_speech_to_scalar(RobotSpeechToken.ASK_READY)
    release = robot_speech_to_scalar(RobotSpeechToken.SAY_RELEASING)

    for step in range(args.steps):
      command._last_update_step = -1
      command.pre_reward_update()
      public_obs = mdp.world_model_public_obs(
        env,
        object_name="manipulation_object",
        command_name="handover",
      )
      action = _build_action(
        env,
        command,
        policy=args.policy,
        step=step,
        delta_pos_scale=args.delta_pos_scale,
        release_distance=args.release_distance,
        speech_scalars={
          "silence": silence,
          "announce": announce,
          "ask_ready": ask_ready,
          "release": release,
        },
      )
      step_out = env.step(action)
      if len(step_out) == 5:
        _obs, reward, terminated, truncated, _extras = step_out
        done = terminated | truncated
      else:
        _obs, reward, done, _extras = step_out

      command._last_update_step = -1
      command.pre_reward_update()
      arrays["public_obs"].append(_cpu(public_obs))
      arrays["action"].append(_cpu(action))
      arrays["reward"].append(_cpu(reward))
      arrays["done"].append(_cpu(done.float()))
      arrays["human_state_id"].append(_cpu(command.human_state_id))
      arrays["human_readiness"].append(_cpu(command.human_readiness))
      arrays["allostatic_load_total"].append(_cpu(command.allostatic_load_total))
      arrays["phase"].append(_cpu(command.phase))
      arrays["reach_progress"].append(_cpu(command.reach_progress))
  finally:
    env.close()

  stacked = {key: np.stack(values, axis=0) for key, values in arrays.items()}
  args.output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    args.output,
    **stacked,
    task_id=np.asarray(args.task_id),
    seed=np.asarray(args.seed),
    policy=np.asarray(args.policy),
  )
  metadata = {
    "task_id": args.task_id,
    "output": str(args.output.resolve()),
    "policy": args.policy,
    "steps": args.steps,
    "num_envs": args.num_envs,
    "public_obs_dim": int(stacked["public_obs"].shape[-1]),
    "action_dim": int(stacked["action"].shape[-1]),
    "device": device,
  }
  args.output.with_suffix(".json").write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )
  _log_wandb(args, metadata, stacked)
  print(json.dumps(metadata, indent=2, sort_keys=True))


def _build_action(
  env: object,
  command: object,
  *,
  policy: str,
  step: int,
  delta_pos_scale: float,
  release_distance: float,
  speech_scalars: dict[str, float],
) -> torch.Tensor:
  scripted = _scripted_action(
    env,
    command,
    delta_pos_scale=delta_pos_scale,
    release_distance=release_distance,
    speech_scalars=speech_scalars,
  )
  if policy == "scripted":
    return scripted
  if policy == "random":
    return torch.empty_like(scripted).uniform_(-1.0, 1.0)
  if policy == "excessive_speech":
    action = torch.empty_like(scripted).uniform_(-0.2, 0.2)
    action[:, 3] = -1.0
    speech = speech_scalars["ask_ready"] if step % 2 == 0 else speech_scalars["announce"]
    action[:, 4] = speech
    return action

  random_action = torch.empty_like(scripted).uniform_(-1.0, 1.0)
  excessive = torch.empty_like(scripted).uniform_(-0.2, 0.2)
  excessive[:, 3] = -1.0
  excessive[:, 4] = (
    speech_scalars["ask_ready"] if step % 2 == 0 else speech_scalars["announce"]
  )
  env_ids = torch.arange(scripted.shape[0], device=scripted.device)
  action = scripted.clone()
  action[env_ids % 3 == 1] = random_action[env_ids % 3 == 1]
  action[env_ids % 3 == 2] = excessive[env_ids % 3 == 2]
  return action


def _scripted_action(
  env: object,
  command: object,
  *,
  delta_pos_scale: float,
  release_distance: float,
  speech_scalars: dict[str, float],
) -> torch.Tensor:
  robot = env.scene.entities["robot"]
  ee_pos = robot.data.site_pos_w[:, command._grasp_site_id, :]
  obj_pos = command.object.data.root_link_pos_w
  object_target = obj_pos + torch.tensor(
    [0.0, 0.0, 0.04],
    dtype=ee_pos.dtype,
    device=ee_pos.device,
  )
  hand_target = command.hand_pos.detach()
  target = torch.where(command.robot_object_grasped.unsqueeze(-1), hand_target, object_target)
  xyz = torch.clamp((target - ee_pos) / delta_pos_scale, -1.0, 1.0)
  should_release = (
    command.robot_object_grasped
    & (command.palm_distance <= release_distance)
    & (command.reach_progress >= 0.1)
  ).unsqueeze(-1)
  gripper = torch.where(
    should_release,
    torch.ones(xyz.shape[0], 1, device=xyz.device, dtype=xyz.dtype),
    -torch.ones(xyz.shape[0], 1, device=xyz.device, dtype=xyz.dtype),
  )
  speech = torch.full(
    (xyz.shape[0], 1),
    speech_scalars["silence"],
    dtype=xyz.dtype,
    device=xyz.device,
  )
  needs_cue = command.human_readiness < command.cfg.readiness_threshold
  speech = torch.where(
    needs_cue.unsqueeze(-1),
    torch.full_like(speech, speech_scalars["announce"]),
    speech,
  )
  speech = torch.where(
    should_release,
    torch.full_like(speech, speech_scalars["release"]),
    speech,
  )
  return torch.cat([xyz, gripper, speech], dim=-1)


def _cpu(tensor: torch.Tensor) -> np.ndarray:
  return tensor.detach().cpu().numpy()


def _log_wandb(
  args: argparse.Namespace,
  metadata: dict[str, object],
  stacked: dict[str, np.ndarray],
) -> None:
  if args.wandb_mode == "disabled":
    return
  import wandb

  run = wandb.init(
    project=args.wandb_project,
    group=args.wandb_group,
    name=args.wandb_run_name,
    mode=args.wandb_mode,
    config=metadata,
  )
  try:
    done = stacked["done"].astype(bool)
    speech = stacked["action"][..., 4]
    metrics = {
      "world_model_dataset/steps": float(metadata["steps"]),
      "world_model_dataset/num_envs": float(metadata["num_envs"]),
      "world_model_dataset/public_obs_dim": float(metadata["public_obs_dim"]),
      "world_model_dataset/action_dim": float(metadata["action_dim"]),
      "world_model_dataset/done_ratio": float(done.mean()),
      "world_model_dataset/reward_mean": float(stacked["reward"].mean()),
      "world_model_dataset/readiness_mean": float(stacked["human_readiness"].mean()),
      "world_model_dataset/load_mean": float(stacked["allostatic_load_total"].mean()),
      "world_model_dataset/speech_abs_mean": float(np.abs(speech).mean()),
    }
    run.log(metrics)
  finally:
    run.finish()


if __name__ == "__main__":
  main()
