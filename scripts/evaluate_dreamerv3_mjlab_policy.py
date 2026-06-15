#!/usr/bin/env python3
"""Evaluate an exact DreamerV3 Mjlab policy checkpoint headlessly."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from allostatic_handover.dreamerv3_exact.dependencies import DEFAULT_DREAMERV3_PATH
from allostatic_handover.dreamerv3_exact.mjlab_bridge import (
  ACTION_KEY,
  MjlabDreamerBridge,
  MjlabDreamerBridgeConfig,
  make_dreamer_spaces,
  make_policy_obs,
)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", required=True, type=Path)
  parser.add_argument("--dreamerv3-path", default=DEFAULT_DREAMERV3_PATH)
  parser.add_argument("--task-id", default=None)
  parser.add_argument("--output", type=Path, default=Path("outputs/dreamerv3_mjlab_policy/eval.json"))
  parser.add_argument("--episodes", type=int, default=64)
  parser.add_argument("--num-envs", type=int, default=None)
  parser.add_argument("--device", default=None)
  parser.add_argument("--seed", type=int, default=None)
  parser.add_argument("--jax-platform", default=None)
  parser.add_argument("--configs", nargs="*", default=None)
  args = parser.parse_args()

  sys.path.insert(0, args.dreamerv3_path)
  sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

  from allostatic_handover.dreamerv3_exact.agent import AllostaticDreamerAgent

  with args.checkpoint.open("rb") as file:
    payload = pickle.load(file)
  ckpt_config = dict(payload.get("config", {}))
  task_id = args.task_id or ckpt_config.get(
    "task_id",
    "Mjlab-Allostatic-Handover-Full-DreamerV3Allostatic",
  )
  num_envs = int(args.num_envs or ckpt_config.get("num_envs", 16))
  device = args.device or ckpt_config.get("device", "cpu")
  seed = int(args.seed if args.seed is not None else ckpt_config.get("seed", 101))

  bridge = MjlabDreamerBridge(
    MjlabDreamerBridgeConfig(
      task_id=task_id,
      num_envs=num_envs,
      seed=seed,
      device=device,
    )
  )
  try:
    obs_space, act_space = make_dreamer_spaces(
      public_obs_dim=bridge.public_obs_dim,
      action_dim=bridge.action_dim,
    )
    agent_config = _make_agent_config(
      ckpt_config,
      args=args,
      fallback_logdir=str(args.output.parent),
    )
    agent = AllostaticDreamerAgent(obs_space, act_space, agent_config)
    agent.load(payload["agent"])
    policy_carry = agent.init_policy(num_envs)
    current_step = bridge.current_step
    completed = 0
    score = np.zeros(num_envs, dtype=np.float64)
    length = np.zeros(num_envs, dtype=np.int64)
    episode_records: list[dict[str, float]] = []
    step_agg: dict[str, list[float]] = defaultdict(list)

    while completed < args.episodes:
      policy_carry, action, _outs = agent.policy(
        policy_carry,
        make_policy_obs(current_step),
        mode="eval",
      )
      current_step = bridge.step(action[ACTION_KEY])
      score += current_step["reward"].astype(np.float64)
      length += 1
      metrics = bridge.current_metrics()
      for key, value in metrics.items():
        step_agg[key].append(float(value))
      done = current_step["is_last"].astype(bool)
      done_ids = np.nonzero(done)[0]
      for env_id in done_ids:
        episode_records.append(
          {
            "score": float(score[env_id]),
            "length": float(length[env_id]),
          }
        )
        score[env_id] = 0.0
        length[env_id] = 0
        completed += 1
        if completed >= args.episodes:
          break

    summary = _summarize(episode_records, step_agg)
    summary.update(
      {
        "checkpoint": str(args.checkpoint.resolve()),
        "task_id": task_id,
        "episodes": int(args.episodes),
        "num_envs": int(num_envs),
        "device": str(device),
      }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
  finally:
    bridge.close()


def _make_agent_config(
  ckpt_config: dict[str, Any],
  *,
  args: argparse.Namespace,
  fallback_logdir: str,
):
  import elements
  import ruamel.yaml as yaml

  dreamerv3_path = args.dreamerv3_path
  configs = elements.Path(Path(dreamerv3_path) / "dreamerv3" / "configs.yaml").read()
  raw = yaml.YAML(typ="safe").load(configs)
  names = args.configs or ckpt_config.get("configs", ("size1m",))
  config = elements.Config(raw["defaults"])
  for name in names:
    config = config.update(raw[name])
  jax_platform = args.jax_platform or ckpt_config.get("jax_platform", "cpu")
  config = config.update(
    logdir=ckpt_config.get("logdir", fallback_logdir),
    seed=int(ckpt_config.get("seed", 101)),
    batch_size=int(ckpt_config.get("batch_size", 16)),
    batch_length=int(ckpt_config.get("batch_length", 64)),
    replay_context=int(ckpt_config.get("replay_context", 0)),
    report_length=min(int(ckpt_config.get("batch_length", 64)), 32),
    consec_train=1,
    consec_report=1,
    jax={
      **config.jax,
      "platform": jax_platform,
      "prealloc": False,
      "expect_devices": 0,
      "enable_policy": True,
    },
  )
  jax_config = elements.Config(
    **{key: value for key, value in config.jax.items()},
    precompile=False,
  )
  return elements.Config(
    **config.agent,
    logdir=config.logdir,
    seed=config.seed,
    jax=jax_config,
    batch_size=config.batch_size,
    batch_length=config.batch_length,
    replay_context=config.replay_context,
    report_length=config.report_length,
    replica=config.replica,
    replicas=config.replicas,
    num_human_states=6,
    aux_state_head={"layers": 1, "units": 128, "act": "silu", "norm": "rms"},
    aux_scalar_head={"layers": 1, "units": 128, "act": "silu", "norm": "rms"},
  )


def _summarize(
  episodes: list[dict[str, float]],
  step_agg: dict[str, list[float]],
) -> dict[str, float]:
  summary: dict[str, float] = {}
  if episodes:
    summary["episode/score_mean"] = float(np.mean([ep["score"] for ep in episodes]))
    summary["episode/length_mean"] = float(np.mean([ep["length"] for ep in episodes]))
  for key, values in step_agg.items():
    if values:
      summary[f"{key}_mean"] = float(np.mean(values))
      summary[f"{key}_max"] = float(np.max(values))
  return summary


if __name__ == "__main__":
  main()
