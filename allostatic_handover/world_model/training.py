"""Training loop for the allostatic belief world model."""

from __future__ import annotations

from pathlib import Path
from typing import TextIO

import json
import sys

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .belief_model import (
  DEFAULT_DREAMERV3_PATH,
  BeliefEstimator,
  BeliefModelConfig,
  save_belief_model,
)
from .dataset import (
  WorldModelSequenceDataset,
  compute_normalization,
  dataset_metadata,
  load_world_model_arrays,
)


def train_belief_world_model(
  dataset_path: str | Path,
  output_dir: str | Path,
  *,
  updates: int = 100,
  batch_size: int = 32,
  seq_len: int = 32,
  lr: float = 3e-4,
  hidden_dim: int = 96,
  belief_dim: int = 16,
  device: str | None = None,
  dreamerv3_path: str = DEFAULT_DREAMERV3_PATH,
  wandb_mode: str = "disabled",
  wandb_project: str = "allostatic-handover-mjlab",
  wandb_run_name: str | None = None,
) -> dict[str, float]:
  """Train a recurrent world-model belief estimator from collected rollouts."""
  arrays = load_world_model_arrays(dataset_path)
  normalization = compute_normalization(arrays)
  config = BeliefModelConfig(
    public_obs_dim=int(arrays["public_obs"].shape[-1]),
    action_dim=int(arrays["action"].shape[-1]),
    hidden_dim=hidden_dim,
    belief_dim=belief_dim,
  )
  dataset = WorldModelSequenceDataset(arrays, seq_len=seq_len, stride=max(1, seq_len // 2))
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
  if len(loader) == 0:
    raise RuntimeError("world-model dataset is empty")

  torch_device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
  model = BeliefEstimator(config, normalization).to(torch_device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  metrics_path = output_dir / "metrics.jsonl"
  metrics: dict[str, float] = {}
  wandb_run = _start_wandb(
    mode=wandb_mode,
    project=wandb_project,
    run_name=wandb_run_name,
    config={
      "dataset_path": str(Path(dataset_path).resolve()),
      "updates": updates,
      "batch_size": batch_size,
      "seq_len": seq_len,
      "lr": lr,
      "hidden_dim": hidden_dim,
      "belief_dim": belief_dim,
      "device": str(torch_device),
      "dreamerv3_path": dreamerv3_path,
    },
  )

  try:
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
      iterator = iter(loader)
      for update in range(1, updates + 1):
        try:
          batch = next(iterator)
        except StopIteration:
          iterator = iter(loader)
          batch = next(iterator)
        batch = {key: value.to(torch_device) for key, value in batch.items()}
        outputs = model.forward_sequence(batch["public_obs"], batch["action"], batch["done"])
        logits = outputs["human_state_logits"]
        state_loss = F.cross_entropy(
          logits.reshape(-1, logits.shape[-1]),
          batch["human_state_id"].reshape(-1).clamp(min=0, max=logits.shape[-1] - 1),
        )
        readiness_loss = F.mse_loss(
          outputs["readiness"].squeeze(-1),
          batch["human_readiness"],
        )
        load_loss = F.mse_loss(
          outputs["load"].squeeze(-1),
          batch["allostatic_load_total"],
        )
        loss = state_loss + readiness_loss + 0.2 * load_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        with torch.no_grad():
          pred_state = logits.argmax(dim=-1)
          state_acc = (pred_state == batch["human_state_id"]).float().mean()
          readiness_mae = torch.abs(
            outputs["readiness"].squeeze(-1) - batch["human_readiness"]
          ).mean()
          load_mae = torch.abs(
            outputs["load"].squeeze(-1) - batch["allostatic_load_total"]
          ).mean()
        metrics = {
          "update": float(update),
          "loss": float(loss.detach().cpu()),
          "human_state_acc": float(state_acc.cpu()),
          "readiness_mae": float(readiness_mae.cpu()),
          "load_mae": float(load_mae.cpu()),
          "state_loss": float(state_loss.detach().cpu()),
          "readiness_loss": float(readiness_loss.detach().cpu()),
          "load_loss": float(load_loss.detach().cpu()),
        }
        _write_jsonl(metrics_file, metrics)
        if wandb_run is not None:
          wandb_run.log({f"world_model/{key}": value for key, value in metrics.items()})
  finally:
    if wandb_run is not None:
      wandb_run.finish()

  extra = {
    "dataset_path": str(Path(dataset_path).resolve()),
    "dataset": dataset_metadata(arrays),
    "dreamerv3": _dreamerv3_status(dreamerv3_path),
  }
  save_belief_model(
    output_dir / "belief_distill.pt",
    model,
    normalization,
    metrics=metrics,
    dreamerv3_path=dreamerv3_path,
    extra=extra,
  )
  save_belief_model(
    output_dir / "world_model.ckpt",
    model,
    normalization,
    metrics=metrics,
    dreamerv3_path=dreamerv3_path,
    extra={**extra, "artifact_role": "dreamer_style_world_model_source"},
  )
  (output_dir / "normalization.json").write_text(
    json.dumps(normalization, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )
  return metrics


def _write_jsonl(file: TextIO, payload: dict[str, float]) -> None:
  file.write(json.dumps(payload, sort_keys=True) + "\n")
  file.flush()


def _dreamerv3_status(dreamerv3_path: str) -> dict[str, object]:
  path = Path(dreamerv3_path)
  status: dict[str, object] = {
    "path": str(path),
    "repo_present": path.exists(),
    "rssm_importable": False,
    "import_error": "",
  }
  if path.exists():
    sys.path.insert(0, str(path))
  try:
    import dreamerv3.rssm  # noqa: F401
  except Exception as exc:
    status["import_error"] = f"{type(exc).__name__}: {exc}"
    return status
  status["rssm_importable"] = True
  return status


def _start_wandb(
  *,
  mode: str,
  project: str,
  run_name: str | None,
  config: dict[str, object],
) -> object | None:
  if mode == "disabled":
    return None
  import wandb

  return wandb.init(
    project=project,
    name=run_name,
    mode=mode,
    config=config,
  )
