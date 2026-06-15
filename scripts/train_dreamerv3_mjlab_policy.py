#!/usr/bin/env python3
"""Train an exact DreamerV3 actor-critic policy on the Mjlab handover task."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from allostatic_handover.dreamerv3_exact.dependencies import DEFAULT_DREAMERV3_PATH
from allostatic_handover.dreamerv3_exact.mjlab_bridge import (
  ACTION_KEY,
  MjlabDreamerBridgeConfig,
  MjlabDreamerBridge,
  checkpoint_payload,
  make_dreamer_spaces,
  make_policy_obs,
  make_replay_stream,
)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dreamerv3-path", default=DEFAULT_DREAMERV3_PATH)
  parser.add_argument(
    "--task-id",
    default="Mjlab-Allostatic-Handover-Full-DreamerV3Allostatic",
  )
  parser.add_argument(
    "--logdir",
    type=Path,
    default=Path("outputs/dreamerv3_mjlab_policy/latest"),
  )
  parser.add_argument("--steps", type=int, default=500_000)
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--seed", type=int, default=101)
  parser.add_argument("--configs", nargs="*", default=("size1m",))
  parser.add_argument("--jax-platform", default="cpu")
  parser.add_argument("--jax-prealloc", default="False")
  parser.add_argument("--jax-precompile", action="store_true")
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--batch-length", type=int, default=64)
  parser.add_argument("--replay-context", type=int, default=0)
  parser.add_argument("--replay-size", type=int, default=1_000_000)
  parser.add_argument("--train-ratio", type=float, default=32.0)
  parser.add_argument("--prefill-steps", type=int, default=2_000)
  parser.add_argument("--log-every", type=int, default=2_000)
  parser.add_argument("--save-every", type=int, default=50_000)
  parser.add_argument("--checkpoint", type=Path, default=None)
  parser.add_argument(
    "--wandb-mode",
    choices=("disabled", "offline", "online"),
    default="disabled",
  )
  parser.add_argument("--wandb-project", default="allostatic-handover-mjlab")
  parser.add_argument("--wandb-group", default=None)
  parser.add_argument("--wandb-run-name", default=None)
  args = parser.parse_args()

  sys.path.insert(0, args.dreamerv3_path)
  sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

  import embodied
  import elements
  from allostatic_handover.dreamerv3_exact.agent import AllostaticDreamerAgent

  args.logdir.mkdir(parents=True, exist_ok=True)
  replay_dir = args.logdir / "replay"
  replay_dir.mkdir(parents=True, exist_ok=True)

  bridge = MjlabDreamerBridge(
    MjlabDreamerBridgeConfig(
      task_id=args.task_id,
      num_envs=args.num_envs,
      seed=args.seed,
      device=args.device,
    )
  )
  try:
    obs_space, act_space = make_dreamer_spaces(
      public_obs_dim=bridge.public_obs_dim,
      action_dim=bridge.action_dim,
    )
    config = _make_agent_config(args)
    agent = AllostaticDreamerAgent(obs_space, act_space, config)
    if args.checkpoint is not None:
      _load_checkpoint(args.checkpoint, agent)
    policy_carry = agent.init_policy(args.num_envs)
    train_carry = agent.init_train(args.batch_size)

    replay_length = args.batch_length + args.replay_context
    replay = embodied.replay.Replay(
      length=replay_length,
      capacity=args.replay_size,
      directory=elements.Path(replay_dir),
      online=False,
      chunksize=max(1024, replay_length),
      name="dreamerv3_mjlab_policy",
      seed=args.seed,
    )
    stream = None
    wandb_run = _start_wandb(args, bridge)
    metrics_path = args.logdir / "metrics.jsonl"
    start_time = time.monotonic()
    total_steps = 0
    updates = 0
    train_accumulator = 0.0
    latest_train_metrics: dict[str, float] = {}
    current_step = bridge.current_step

    with metrics_path.open("a", encoding="utf-8") as metrics_file:
      while total_steps < args.steps:
        if total_steps < args.prefill_steps:
          action = bridge.random_action()
        else:
          policy_carry, action, _outs = agent.policy(
            policy_carry,
            make_policy_obs(current_step),
            mode="train",
          )
        transition = bridge.transition(current_step, action)
        _add_vector_transition(replay, transition)
        current_step = bridge.step(action[ACTION_KEY])
        total_steps += args.num_envs

        batch_steps = args.batch_size * args.batch_length
        train_accumulator += args.train_ratio * args.num_envs / max(batch_steps, 1)
        if len(replay) > 0 and total_steps >= args.prefill_steps:
          if stream is None:
            stream = iter(
              agent.stream(
                make_replay_stream(
                  replay,
                  batch_size=args.batch_size,
                  batch_length=args.batch_length,
                  replay_context=args.replay_context,
                )
              )
            )
          while train_accumulator >= 1.0:
            batch = next(stream)
            train_carry, outs, metrics = agent.train(train_carry, batch)
            if "replay" in outs:
              replay.update(outs["replay"])
            latest_train_metrics = _to_float_metrics(metrics)
            updates += 1
            train_accumulator -= 1.0

        should_log = (
          total_steps % max(args.log_every, args.num_envs) < args.num_envs
          or total_steps >= args.steps
        )
        should_save = (
          total_steps % max(args.save_every, args.num_envs) < args.num_envs
          or total_steps >= args.steps
        )
        if should_log:
          record = {
            "step": float(total_steps),
            "updates": float(updates),
            "elapsed_s": float(time.monotonic() - start_time),
            "replay/items": float(len(replay)),
            **bridge.current_metrics(),
            **{f"train/{k}": v for k, v in latest_train_metrics.items()},
          }
          metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
          metrics_file.flush()
          print(json.dumps(record, sort_keys=True), flush=True)
          if wandb_run is not None:
            wandb_run.log(record, step=total_steps)

        if should_save:
          _save_checkpoint(
            args.logdir / "policy.ckpt",
            agent=agent,
            args=args,
            bridge=bridge,
            replay=replay,
            replay_dir=replay_dir,
            metrics=latest_train_metrics,
          )
          _save_checkpoint(
            args.logdir / f"policy_step_{total_steps}.ckpt",
            agent=agent,
            args=args,
            bridge=bridge,
            replay=replay,
            replay_dir=replay_dir,
            metrics=latest_train_metrics,
          )
    if wandb_run is not None:
      wandb_run.finish()
  finally:
    bridge.close()


def _make_agent_config(args: argparse.Namespace):
  import elements
  import ruamel.yaml as yaml

  configs = elements.Path(Path(args.dreamerv3_path) / "dreamerv3" / "configs.yaml").read()
  configs = yaml.YAML(typ="safe").load(configs)
  config = elements.Config(configs["defaults"])
  for name in args.configs:
    config = config.update(configs[name])
  config = config.update(
    logdir=str(args.logdir),
    seed=args.seed,
    batch_size=args.batch_size,
    batch_length=args.batch_length,
    replay_context=args.replay_context,
    report_length=min(args.batch_length, 32),
    consec_train=1,
    consec_report=1,
    jax={
      **config.jax,
      "platform": args.jax_platform,
      "prealloc": args.jax_prealloc.lower() in {"1", "true", "yes"},
      "expect_devices": 0,
      "enable_policy": True,
    },
  )
  jax_config = elements.Config(
    **{key: value for key, value in config.jax.items()},
    precompile=bool(args.jax_precompile),
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


def _add_vector_transition(replay: Any, transition: dict[str, np.ndarray]) -> None:
  num_envs = int(next(iter(transition.values())).shape[0])
  for env_id in range(num_envs):
    replay.add({key: value[env_id] for key, value in transition.items()}, worker=env_id)


def _save_checkpoint(
  path: Path,
  *,
  agent: Any,
  args: argparse.Namespace,
  bridge: MjlabDreamerBridge,
  replay: Any,
  replay_dir: Path,
  metrics: dict[str, float],
) -> None:
  replay.save()
  payload = checkpoint_payload(
    agent=agent,
    config=vars(args),
    public_obs_dim=bridge.public_obs_dim,
    action_dim=bridge.action_dim,
    replay_dir=replay_dir,
    metrics=metrics,
  )
  with path.open("wb") as file:
    pickle.dump(payload, file)


def _load_checkpoint(path: Path, agent: Any) -> None:
  with path.open("rb") as file:
    payload = pickle.load(file)
  agent.load(payload["agent"])


def _to_float_metrics(metrics: dict[str, object]) -> dict[str, float]:
  result: dict[str, float] = {}
  for key, value in metrics.items():
    array = np.asarray(value)
    if array.shape == ():
      result[key] = float(array)
  return result


def _start_wandb(args: argparse.Namespace, bridge: MjlabDreamerBridge):
  if args.wandb_mode == "disabled":
    return None
  import wandb

  return wandb.init(
    project=args.wandb_project,
    group=args.wandb_group,
    name=args.wandb_run_name,
    mode=args.wandb_mode,
    config={
      **vars(args),
      "public_obs_dim": bridge.public_obs_dim,
      "action_dim": bridge.action_dim,
    },
  )


if __name__ == "__main__":
  main()
