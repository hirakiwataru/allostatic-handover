#!/usr/bin/env python3
"""Train an exact DreamerV3 RSSM world model with allostatic auxiliary heads."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

from allostatic_handover.dreamerv3_exact.dataset import (
  DreamerBatchConfig,
  OfflineWorldModelBatchStream,
)
from allostatic_handover.dreamerv3_exact.dependencies import DEFAULT_DREAMERV3_PATH


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dataset", required=True, type=Path)
  parser.add_argument("--output-dir", required=True, type=Path)
  parser.add_argument("--dreamerv3-path", default=DEFAULT_DREAMERV3_PATH)
  parser.add_argument("--updates", type=int, default=100)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--batch-length", type=int, default=32)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--configs", nargs="*", default=("debug",))
  parser.add_argument("--jax-platform", default="cpu")
  parser.add_argument("--jax-prealloc", default="False")
  parser.add_argument("--wandb-mode", default="disabled")
  parser.add_argument("--wandb-project", default="allostatic-handover-mjlab")
  parser.add_argument("--wandb-run-name", default=None)
  args = parser.parse_args()

  sys.path.insert(0, args.dreamerv3_path)
  sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

  import elements
  import ruamel.yaml as yaml
  from allostatic_handover.dreamerv3_exact.agent import (
    AllostaticDreamerAgent,
  )

  arrays = np.load(args.dataset, allow_pickle=False)
  public_obs_dim = int(arrays["public_obs"].shape[-1])
  action_dim = int(arrays["action"].shape[-1])
  arrays.close()

  config = _load_dreamer_config(args.dreamerv3_path, args.configs)
  config = config.update(
    logdir=str(args.output_dir),
    seed=args.seed,
    batch_size=args.batch_size,
    batch_length=args.batch_length,
    replay_context=0,
    report_length=min(args.batch_length, 32),
    jax={
      **config.jax,
      "platform": args.jax_platform,
      "prealloc": args.jax_prealloc.lower() in {"1", "true", "yes"},
      "expect_devices": 0,
      "enable_policy": False,
    },
  )
  jax_config = elements.Config(
    **{key: value for key, value in config.jax.items()},
    precompile=False,
  )
  agent_config = elements.Config(
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

  obs_space = {
    "public_obs": elements.Space(np.float32, (public_obs_dim,)),
    "reward": elements.Space(np.float32, ()),
    "is_first": elements.Space(bool, (), 0, 2),
    "is_last": elements.Space(bool, (), 0, 2),
    "is_terminal": elements.Space(bool, (), 0, 2),
  }
  act_space = {
    "action": elements.Space(np.float32, (action_dim,), -1.0, 1.0),
  }
  args.output_dir.mkdir(parents=True, exist_ok=True)
  config.save(elements.Path(args.output_dir / "config.yaml"))

  stream = OfflineWorldModelBatchStream(
    args.dataset,
    DreamerBatchConfig(
      batch_size=args.batch_size,
      batch_length=args.batch_length,
      replay_context=0,
      seed=args.seed,
    ),
  )
  agent = AllostaticDreamerAgent(obs_space, act_space, agent_config)
  carry = agent.init_train(args.batch_size)
  train_stream = iter(agent.stream(stream))
  wandb_run = _start_wandb(args)
  metrics_path = args.output_dir / "metrics.jsonl"
  latest_metrics: dict[str, float] = {}
  try:
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
      for update in range(1, args.updates + 1):
        batch = next(train_stream)
        carry, _outs, metrics = agent.train(carry, batch)
        latest_metrics = _to_float_metrics(metrics)
        latest_metrics["update"] = float(update)
        metrics_file.write(json.dumps(latest_metrics, sort_keys=True) + "\n")
        metrics_file.flush()
        if wandb_run is not None:
          wandb_run.log({f"dreamerv3_exact/{k}": v for k, v in latest_metrics.items()})
  finally:
    if wandb_run is not None:
      wandb_run.finish()

  payload = {
    "format": "allostatic_handover_exact_dreamerv3_v1",
    "dreamerv3_path": args.dreamerv3_path,
    "dataset": str(args.dataset.resolve()),
    "obs_space": {"public_obs_dim": public_obs_dim},
    "act_space": {"action_dim": action_dim},
    "runtime_belief_note": (
      "This is the exact DreamerV3/JAX RSSM checkpoint. Mjlab PPO uses the "
      "separate PyTorch belief_distill.pt artifact because RSL-RL runs in "
      "PyTorch and should not call JAX inside every environment step."
    ),
    "agent": agent.save(),
    "metrics": latest_metrics,
  }
  with (args.output_dir / "world_model.ckpt").open("wb") as file:
    pickle.dump(payload, file)
  print(json.dumps(latest_metrics, indent=2, sort_keys=True))


def _load_dreamer_config(dreamerv3_path: str, names: tuple[str, ...] | list[str]):
  import elements
  import ruamel.yaml as yaml

  configs = elements.Path(Path(dreamerv3_path) / "dreamerv3" / "configs.yaml").read()
  configs = yaml.YAML(typ="safe").load(configs)
  config = elements.Config(configs["defaults"])
  for name in names:
    config = config.update(configs[name])
  return config


def _to_float_metrics(metrics: dict[str, object]) -> dict[str, float]:
  result: dict[str, float] = {}
  for key, value in metrics.items():
    array = np.asarray(value)
    if array.shape == ():
      result[key] = float(array)
  return result


def _start_wandb(args: argparse.Namespace):
  if args.wandb_mode == "disabled":
    return None
  import wandb

  return wandb.init(
    project=args.wandb_project,
    name=args.wandb_run_name,
    mode=args.wandb_mode,
    config=vars(args),
  )


if __name__ == "__main__":
  main()
