#!/usr/bin/env python3
"""Scripted feasibility check for Mjlab Full handover tasks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "task_id",
    nargs="?",
    default="Mjlab-Allostatic-Handover-Full-TaskOnly",
  )
  parser.add_argument("--device", default=None)
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument("--steps", type=int, default=160)
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument("--release-distance", type=float, default=0.30)
  parser.add_argument(
    "--min-release-progress",
    type=float,
    default=0.0,
    help="Require the human reach-out progress to reach this value before opening the gripper.",
  )
  parser.add_argument("--delta-pos-scale", type=float, default=0.045)
  parser.add_argument(
    "--robot-root-offset",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.0),
    metavar=("DX", "DY", "DZ"),
    help="Diagnostic offset applied to the configured Yam root position before env creation.",
  )
  parser.add_argument("--progress-interval", type=int, default=20)
  parser.add_argument("--trace-distance", type=float, default=0.40)
  parser.add_argument("--trace-limit", type=int, default=40)
  parser.add_argument(
    "--auto-reset",
    action="store_true",
    help="Keep Mjlab auto-reset enabled. By default the terminal state is preserved for diagnosis.",
  )
  parser.add_argument("--fail-on-no-success", action="store_true")
  parser.add_argument("--output", type=Path, default=None)
  args = parser.parse_args()

  os.environ.setdefault("MUJOCO_GL", "egl")
  os.environ.setdefault("XDG_CACHE_HOME", "/mnt/k_iwamoto/sim_data/tmp/xdg_cache")
  os.environ.setdefault("MPLCONFIGDIR", "/mnt/k_iwamoto/sim_data/tmp/matplotlib")

  import allostatic_handover.mjlab_tasks  # noqa: F401
  from allostatic_handover.envs.speech_events import (
    RobotSpeechToken,
    robot_speech_to_scalar,
  )
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg
  from mjlab.utils.torch import configure_torch_backends

  configure_torch_backends()
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  cfg = load_env_cfg(args.task_id, play=True)
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  cfg.auto_reset = args.auto_reset
  if any(abs(v) > 0.0 for v in args.robot_root_offset):
    robot = cfg.scene.entities["robot"]
    robot.init_state.pos = tuple(
      float(a) + float(b) for a, b in zip(robot.init_state.pos, args.robot_root_offset)
    )
  env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)

  progress: list[dict[str, float | int]] = []
  trace: list[dict[str, float | int | bool]] = []
  try:
    env.reset()
    command = env.command_manager.get_term("handover")
    robot = env.scene.entities["robot"]
    grasp_id = command._grasp_site_id
    joint_ids = getattr(env.action_manager.get_term("arm_ik"), "_joint_ids", None)

    command._last_update_step = -1
    command.pre_reward_update()
    initial_distance = command.palm_distance.detach().clone()
    min_distance = initial_distance.clone()
    min_step = torch.zeros(args.num_envs, dtype=torch.long, device=device)
    min_ee_pos = robot.data.site_pos_w[:, grasp_id, :].detach().clone()
    min_hand_pos = command.hand_pos.detach().clone()
    min_object_pos = command.object.data.root_link_pos_w.detach().clone()
    max_success = command.episode_success.detach().clone()
    max_attached = command.object_attached.detach().clone().float()
    max_release_at_hand = command.release_at_hand_event.detach().clone()
    max_robot_reached_hand = command.robot_reached_hand.detach().clone().float()
    max_can_release = torch.zeros(args.num_envs, device=device)
    done_ever = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    terminated_ever = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    truncated_ever = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    max_reward = torch.full((args.num_envs,), -float("inf"), device=device)
    min_distance_while_release = torch.full_like(initial_distance, float("inf"))
    stopped_step = args.steps - 1

    last_reward = torch.zeros(args.num_envs, device=device)
    last_done = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    for step in range(args.steps):
      ee_pos = robot.data.site_pos_w[:, grasp_id, :]
      target = command.hand_pos.detach()
      error = target - ee_pos
      xyz_action = torch.clamp(error / args.delta_pos_scale, -1.0, 1.0)
      should_release = (
        (command.palm_distance <= args.release_distance)
        & (command.reach_progress >= args.min_release_progress)
      ).unsqueeze(-1)
      gripper = torch.where(
        should_release,
        torch.ones(args.num_envs, 1, device=device),
        -torch.ones(args.num_envs, 1, device=device),
      )
      action_parts = [xyz_action, gripper]
      if "speech" in env.action_manager.active_terms:
        announce_scalar = robot_speech_to_scalar(RobotSpeechToken.ANNOUNCE_HANDOVER)
        release_scalar = robot_speech_to_scalar(RobotSpeechToken.SAY_RELEASING)
        silence_scalar = robot_speech_to_scalar(RobotSpeechToken.SILENCE)
        speech = torch.full(
          (args.num_envs, 1),
          silence_scalar,
          dtype=xyz_action.dtype,
          device=device,
        )
        speech = torch.where(
          (command.reach_progress < args.min_release_progress).unsqueeze(-1),
          torch.full_like(speech, announce_scalar),
          speech,
        )
        speech = torch.where(
          should_release,
          torch.full_like(speech, release_scalar),
          speech,
        )
        action_parts.append(speech)
      action = torch.cat(action_parts, dim=-1)

      step_out = env.step(action)
      if len(step_out) == 5:
        _obs, reward, terminated, truncated, _extras = step_out
        done = terminated | truncated
        terminated_ever |= terminated.detach().bool()
        truncated_ever |= truncated.detach().bool()
      else:
        _obs, reward, done, _extras = step_out
        terminated_ever |= done.detach().bool()
      last_reward = reward.detach()
      last_done = done.detach().bool()
      done_ever |= last_done
      max_reward = torch.maximum(max_reward, last_reward)
      improved = command.palm_distance.detach() < min_distance
      min_distance = torch.minimum(min_distance, command.palm_distance.detach())
      if improved.any():
        min_step[improved] = step
        min_ee_pos[improved] = robot.data.site_pos_w[:, grasp_id, :].detach()[improved]
        min_hand_pos[improved] = command.hand_pos.detach()[improved]
        min_object_pos[improved] = command.object.data.root_link_pos_w.detach()[improved]
      max_success = torch.maximum(max_success, command.episode_success.detach())
      max_attached = torch.maximum(max_attached, command.object_attached.detach().float())
      max_release_at_hand = torch.maximum(
        max_release_at_hand,
        command.release_at_hand_event.detach(),
      )
      can_release_now = command._gripper_release_signal().detach()
      max_can_release = torch.maximum(max_can_release, can_release_now.float())
      max_robot_reached_hand = torch.maximum(
        max_robot_reached_hand,
        command.robot_reached_hand.detach().float(),
      )
      min_distance_while_release = torch.where(
        can_release_now,
        torch.minimum(min_distance_while_release, command.palm_distance.detach()),
        min_distance_while_release,
      )

      if (
        args.progress_interval > 0
        and (step % args.progress_interval == 0 or step == args.steps - 1)
      ):
        progress.append(
          {
            "step": step,
            "palm_distance_mean": float(command.palm_distance.mean().detach().cpu()),
            "palm_distance_min": float(command.palm_distance.min().detach().cpu()),
            "ee_to_hand_mean": float(
              torch.norm(
                command.hand_pos.detach() - robot.data.site_pos_w[:, grasp_id, :].detach(),
                dim=-1,
              )
              .mean()
              .cpu()
            ),
            "reach_progress_mean": float(command.reach_progress.mean().detach().cpu()),
            "animation_frame_mean": float(command.animation_frame.mean().detach().cpu()),
            "phase_mean": float(command.phase.float().mean().detach().cpu()),
            "can_release_rate_so_far": float(max_can_release.mean().detach().cpu()),
            "robot_reached_hand_rate_so_far": float(
              max_robot_reached_hand.mean().detach().cpu()
            ),
            "success_rate_so_far": float(max_success.mean().detach().cpu()),
            "attached_rate_so_far": float(max_attached.mean().detach().cpu()),
            "release_at_hand_rate_so_far": float(
              (max_release_at_hand > 0.0).float().mean().detach().cpu()
            ),
              "done_rate_last_step": float(last_done.float().mean().detach().cpu()),
              "done_rate_ever": float(done_ever.float().mean().detach().cpu()),
              "max_reward_mean": float(max_reward.mean().detach().cpu()),
          }
        )
      if len(trace) < args.trace_limit:
        env_id = int(torch.argmin(command.palm_distance.detach()).cpu())
        near = float(command.palm_distance[env_id].detach().cpu()) <= args.trace_distance
        interesting = (
          near
          or bool(can_release_now[env_id].detach().cpu())
          or bool(command.robot_reached_hand[env_id].detach().cpu())
          or bool(command.object_attached[env_id].detach().cpu())
        )
        if interesting:
          trace.append(
            {
              "step": step,
              "env": env_id,
              "palm_distance": float(command.palm_distance[env_id].detach().cpu()),
              "can_release": bool(can_release_now[env_id].detach().cpu()),
              "robot_reached_hand": bool(
                command.robot_reached_hand[env_id].detach().cpu()
              ),
              "release_at_hand_event": float(
                command.release_at_hand_event[env_id].detach().cpu()
              ),
              "object_attached": bool(command.object_attached[env_id].detach().cpu()),
              "robot_object_grasped": bool(
                command.robot_object_grasped[env_id].detach().cpu()
              ),
              "phase": int(command.phase[env_id].detach().cpu()),
              "animation_frame": float(command.animation_frame[env_id].detach().cpu()),
              "reach_progress": float(command.reach_progress[env_id].detach().cpu()),
              "raw_gripper_action": float(
                env.action_manager.get_term("gripper").raw_action[env_id, 0]
                .detach()
                .cpu()
              ),
              "done": bool(last_done[env_id].detach().cpu()),
            }
          )
      if not args.auto_reset and last_done.any():
        stopped_step = step
        break

    env0 = int(torch.argmin(min_distance).detach().cpu())
    final_ee_pos = robot.data.site_pos_w[:, grasp_id, :].detach()
    final_hand_pos = command.hand_pos.detach()
    final_object_pos = command.object.data.root_link_pos_w.detach()
    joint_diag: dict[str, object] = {}
    if joint_ids is not None:
      joint_pos = robot.data.joint_pos[:, joint_ids].detach()
      limits = robot.data.soft_joint_pos_limits[:, joint_ids, :].detach()
      lower_margin = joint_pos - limits[:, :, 0]
      upper_margin = limits[:, :, 1] - joint_pos
      joint_diag = {
        "joint_limit_min_lower_margin": float(lower_margin.min().cpu()),
        "joint_limit_min_upper_margin": float(upper_margin.min().cpu()),
        "joint_pos_env0": joint_pos[0].cpu().tolist(),
        "joint_lower_env0": limits[0, :, 0].cpu().tolist(),
        "joint_upper_env0": limits[0, :, 1].cpu().tolist(),
      }

    result = {
      "device": device,
      "task_id": args.task_id,
      "num_envs": args.num_envs,
      "steps": args.steps,
      "seed": args.seed,
      "robot_root_offset": list(args.robot_root_offset),
      "auto_reset": args.auto_reset,
      "release_distance": args.release_distance,
      "min_release_progress": args.min_release_progress,
      "stopped_step": stopped_step,
      "initial_palm_distance_mean": float(initial_distance.mean().cpu()),
      "initial_palm_distance_min": float(initial_distance.min().cpu()),
      "min_palm_distance_mean": float(min_distance.mean().cpu()),
      "min_palm_distance_min": float(min_distance.min().cpu()),
      "min_distance_env_index": env0,
      "min_distance_step": int(min_step[env0].cpu()),
      "min_distance_ee_pos": min_ee_pos[env0].cpu().tolist(),
      "min_distance_hand_pos": min_hand_pos[env0].cpu().tolist(),
      "min_distance_object_pos": min_object_pos[env0].cpu().tolist(),
      "min_distance_ee_to_hand": float(
        torch.norm(min_ee_pos[env0] - min_hand_pos[env0]).cpu()
      ),
      "final_ee_pos_env0": final_ee_pos[0].cpu().tolist(),
      "final_hand_pos_env0": final_hand_pos[0].cpu().tolist(),
      "final_object_pos_env0": final_object_pos[0].cpu().tolist(),
      "final_palm_distance_mean": float(command.palm_distance.mean().detach().cpu()),
      "success_rate": float(max_success.mean().detach().cpu()),
      "done_rate_ever": float(done_ever.float().mean().detach().cpu()),
      "terminated_rate_ever": float(terminated_ever.float().mean().detach().cpu()),
      "truncated_rate_ever": float(truncated_ever.float().mean().detach().cpu()),
      "attached_rate": float(max_attached.mean().detach().cpu()),
      "release_at_hand_rate": float(
        (max_release_at_hand > 0.0).float().mean().detach().cpu()
      ),
      "robot_reached_hand_rate": float(max_robot_reached_hand.mean().detach().cpu()),
      "can_release_rate": float(max_can_release.mean().detach().cpu()),
      "min_distance_while_release_mean": float(
        torch.where(
          torch.isfinite(min_distance_while_release),
          min_distance_while_release,
          torch.zeros_like(min_distance_while_release),
        )
        .mean()
        .cpu()
      ),
      "min_distance_while_release_min": float(
        torch.where(
          torch.isfinite(min_distance_while_release),
          min_distance_while_release,
          torch.full_like(min_distance_while_release, 1.0e9),
        )
        .min()
        .cpu()
      ),
      "robot_object_grasped_rate_final": float(
        command.robot_object_grasped.float().mean().detach().cpu()
      ),
      "phase_final_mean": float(command.phase.float().mean().detach().cpu()),
      "last_reward_mean": float(last_reward.mean().cpu()),
      "max_reward_mean": float(max_reward.mean().cpu()),
      "max_reward_max": float(max_reward.max().cpu()),
      "last_done_rate": float(last_done.float().mean().cpu()),
      "joint_diagnostics": joint_diag,
      "progress": progress,
      "trace": trace,
    }
  finally:
    env.close()

  text = json.dumps(result, indent=2, sort_keys=True)
  if args.output is not None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")
  print(text)
  if args.fail_on_no_success and result["success_rate"] <= 0.0:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
