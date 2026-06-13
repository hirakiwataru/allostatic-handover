#!/usr/bin/env python3
"""Headless evaluation for Mjlab RSL-RL checkpoints."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "task_id",
    nargs="?",
    default="Mjlab-Allostatic-Handover-Full-TaskOnly",
  )
  parser.add_argument("--checkpoint-file", required=True)
  parser.add_argument("--episodes", type=int, default=64)
  parser.add_argument("--num-envs", type=int, default=64)
  parser.add_argument("--device", default=None)
  parser.add_argument("--seed", type=int, default=123)
  parser.add_argument("--max-steps", type=int, default=None)
  parser.add_argument("--output", type=Path, default=None)
  args = parser.parse_args()

  os.environ.setdefault("MUJOCO_GL", "egl")
  os.environ.setdefault("XDG_CACHE_HOME", "/mnt/k_iwamoto/sim_data/tmp/xdg_cache")
  os.environ.setdefault("MPLCONFIGDIR", "/mnt/k_iwamoto/sim_data/tmp/matplotlib")

  # Populate the Mjlab registry, including external editable tasks.
  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.utils.torch import configure_torch_backends

  configure_torch_backends()
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(args.task_id, play=False)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed
  agent_cfg = load_rl_cfg(args.task_id)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  try:
    runner_cls = load_runner_cls(args.task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      args.checkpoint_file,
      load_cfg={"actor": True},
      strict=True,
      map_location=device,
    )
    policy = runner.get_inference_policy(device=device)

    obs, _ = env.reset()
    max_steps = args.max_steps or env.max_episode_length * max(1, args.episodes)
    completed = 0
    success_sum = 0.0
    return_sum = 0.0
    reward_sum = torch.zeros(env.num_envs, device=device)
    length_sum = 0.0
    last_log: dict[str, float] = {}

    for _step in range(max_steps):
      with torch.inference_mode():
        action = policy(obs)
      obs, reward, dones, extras = env.step(action)
      reward_sum += reward
      done_count = int(dones.sum().item())
      if done_count == 0:
        continue

      log = extras.get("log", {})
      success = _scalar(log.get("Episode_Metrics/success", 0.0))
      mean_reward = _scalar(log.get("Episode_Reward/total", reward_sum[dones.bool()].mean()))
      mean_length = _scalar(log.get("Episode_Termination/time_out", 0.0))
      success_sum += success * done_count
      return_sum += mean_reward * done_count
      length_sum += mean_length * done_count
      completed += done_count
      reward_sum[dones.bool()] = 0.0
      last_log = {key: _scalar(value) for key, value in log.items()}
      if completed >= args.episodes:
        break

    episodes = max(completed, 1)
    result = {
      "task_id": args.task_id,
      "checkpoint_file": str(Path(args.checkpoint_file).resolve()),
      "requested_episodes": args.episodes,
      "completed_episodes": completed,
      "success_rate": success_sum / episodes,
      "mean_episode_reward": return_sum / episodes,
      "mean_time_out_metric": length_sum / episodes,
      "device": device,
      "num_envs": args.num_envs,
      "seed": args.seed,
      "max_steps": max_steps,
      "last_episode_log": last_log,
    }
  finally:
    env.close()

  text = json.dumps(result, indent=2, sort_keys=True)
  if args.output is not None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")
  print(text)


def _scalar(value: Any) -> float:
  if isinstance(value, torch.Tensor):
    return float(value.detach().mean().cpu())
  return float(value)


if __name__ == "__main__":
  main()
