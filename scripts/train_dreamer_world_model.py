#!/usr/bin/env python3
"""Train the allostatic Dreamer-style world-model belief estimator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from allostatic_handover.world_model.belief_model import DEFAULT_DREAMERV3_PATH
from allostatic_handover.world_model.training import train_belief_world_model


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dataset", required=True, type=Path)
  parser.add_argument("--output-dir", required=True, type=Path)
  parser.add_argument("--updates", type=int, default=100)
  parser.add_argument("--batch-size", type=int, default=32)
  parser.add_argument("--seq-len", type=int, default=32)
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--hidden-dim", type=int, default=96)
  parser.add_argument("--belief-dim", type=int, default=16)
  parser.add_argument("--device", default=None)
  parser.add_argument("--dreamerv3-path", default=DEFAULT_DREAMERV3_PATH)
  parser.add_argument("--wandb-mode", default="disabled")
  parser.add_argument("--wandb-project", default="allostatic-handover-mjlab")
  parser.add_argument("--wandb-run-name", default=None)
  args = parser.parse_args()

  metrics = train_belief_world_model(
    args.dataset,
    args.output_dir,
    updates=args.updates,
    batch_size=args.batch_size,
    seq_len=args.seq_len,
    lr=args.lr,
    hidden_dim=args.hidden_dim,
    belief_dim=args.belief_dim,
    device=args.device,
    dreamerv3_path=args.dreamerv3_path,
    wandb_mode=args.wandb_mode,
    wandb_project=args.wandb_project,
    wandb_run_name=args.wandb_run_name,
  )
  print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
