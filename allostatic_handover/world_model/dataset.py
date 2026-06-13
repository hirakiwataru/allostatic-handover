"""Dataset helpers for offline allostatic world-model training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def load_world_model_arrays(path: str | Path) -> dict[str, np.ndarray]:
  """Load a world-model rollout dataset from ``.npz``."""
  with np.load(path, allow_pickle=False) as data:
    return {key: data[key] for key in data.files}


class WorldModelSequenceDataset(Dataset[dict[str, torch.Tensor]]):
  """Return fixed-length sequences from ``[time, env, ...]`` rollout arrays."""

  def __init__(
    self,
    arrays: dict[str, np.ndarray],
    seq_len: int = 32,
    stride: int | None = None,
  ) -> None:
    if arrays["public_obs"].ndim != 3:
      raise ValueError("public_obs must have shape [time, env, dim]")
    if arrays["action"].shape[:2] != arrays["public_obs"].shape[:2]:
      raise ValueError("action must have the same [time, env] axes as public_obs")
    self.arrays = arrays
    self.seq_len = int(seq_len)
    self.stride = int(stride or seq_len)
    time, num_envs, _ = arrays["public_obs"].shape
    if time < self.seq_len:
      raise ValueError(f"dataset has only {time} steps, shorter than seq_len={seq_len}")
    self.indices = [
      (env_id, start)
      for env_id in range(num_envs)
      for start in range(0, time - self.seq_len + 1, self.stride)
    ]

  def __len__(self) -> int:
    return len(self.indices)

  def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
    env_id, start = self.indices[index]
    stop = start + self.seq_len
    return {
      "public_obs": _float_tensor(self.arrays["public_obs"][start:stop, env_id]),
      "action": _float_tensor(self.arrays["action"][start:stop, env_id]),
      "reward": _float_tensor(self.arrays["reward"][start:stop, env_id]),
      "done": _float_tensor(self.arrays["done"][start:stop, env_id]),
      "human_state_id": _long_tensor(self.arrays["human_state_id"][start:stop, env_id]),
      "human_readiness": _float_tensor(self.arrays["human_readiness"][start:stop, env_id]),
      "allostatic_load_total": _float_tensor(
        self.arrays["allostatic_load_total"][start:stop, env_id]
      ),
      "phase": _long_tensor(self.arrays["phase"][start:stop, env_id]),
      "reach_progress": _float_tensor(self.arrays["reach_progress"][start:stop, env_id]),
    }


def compute_normalization(arrays: dict[str, np.ndarray]) -> dict[str, list[float]]:
  """Compute input statistics for public observations and actions."""
  public_obs = arrays["public_obs"].reshape(-1, arrays["public_obs"].shape[-1])
  action = arrays["action"].reshape(-1, arrays["action"].shape[-1])
  return {
    "public_obs_mean": public_obs.mean(axis=0).astype(np.float32).tolist(),
    "public_obs_std": (public_obs.std(axis=0) + 1e-6).astype(np.float32).tolist(),
    "action_mean": action.mean(axis=0).astype(np.float32).tolist(),
    "action_std": (action.std(axis=0) + 1e-6).astype(np.float32).tolist(),
  }


def dataset_metadata(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
  """Return small JSON-serializable metadata for a dataset."""
  time, num_envs, public_obs_dim = arrays["public_obs"].shape
  return {
    "time_steps": int(time),
    "num_envs": int(num_envs),
    "public_obs_dim": int(public_obs_dim),
    "action_dim": int(arrays["action"].shape[-1]),
    "num_samples": int(time * num_envs),
  }


def _float_tensor(array: np.ndarray) -> torch.Tensor:
  return torch.as_tensor(array, dtype=torch.float32)


def _long_tensor(array: np.ndarray) -> torch.Tensor:
  return torch.as_tensor(array, dtype=torch.long)
