"""Offline dataset streaming for exact DreamerV3 training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DreamerBatchConfig:
  batch_size: int = 16
  batch_length: int = 32
  replay_context: int = 0
  seed: int = 0

  @property
  def sequence_length(self) -> int:
    return self.batch_length + self.replay_context


class OfflineWorldModelBatchStream:
  """Infinite stream of DreamerV3-compatible batches from a rollout ``.npz``."""

  def __init__(
    self,
    dataset_path: str | Path,
    config: DreamerBatchConfig,
  ) -> None:
    self.dataset_path = Path(dataset_path)
    self.config = config
    with np.load(self.dataset_path, allow_pickle=False) as data:
      self.arrays = {key: data[key] for key in data.files}
    self.public_obs = _require(self.arrays, "public_obs").astype(np.float32)
    self.action = _require(self.arrays, "action").astype(np.float32)
    self.reward = _require(self.arrays, "reward").astype(np.float32)
    self.done = _require(self.arrays, "done").astype(bool)
    self.human_state_id = _require(self.arrays, "human_state_id").astype(np.int32)
    self.human_readiness = _require(self.arrays, "human_readiness").astype(np.float32)
    self.allostatic_load_total = _require(
      self.arrays,
      "allostatic_load_total",
    ).astype(np.float32)
    self.time, self.num_envs, self.public_obs_dim = self.public_obs.shape
    self.action_dim = self.action.shape[-1]
    if self.time < config.sequence_length:
      raise ValueError(
        f"dataset has {self.time} steps but requires {config.sequence_length}"
      )
    self._rng = np.random.default_rng(config.seed)

  def __iter__(self) -> "OfflineWorldModelBatchStream":
    return self

  def __next__(self) -> dict[str, np.ndarray]:
    starts = self._rng.integers(
      0,
      self.time - self.config.sequence_length + 1,
      size=self.config.batch_size,
    )
    env_ids = self._rng.integers(0, self.num_envs, size=self.config.batch_size)
    seqs = [
      self._sequence(env_id=int(env_id), start=int(start))
      for env_id, start in zip(env_ids, starts)
    ]
    return {
      key: np.stack([seq[key] for seq in seqs], axis=0)
      for key in seqs[0]
    }

  def _sequence(self, env_id: int, start: int) -> dict[str, np.ndarray]:
    stop = start + self.config.sequence_length
    done = self.done[start:stop, env_id]
    is_first = np.zeros_like(done, dtype=bool)
    is_first[0] = True
    if len(done) > 1:
      is_first[1:] = done[:-1]
    stepid = np.zeros((self.config.sequence_length, 20), dtype=np.uint8)
    stepid[:, :8] = np.arange(start, stop, dtype=np.uint64)[:, None].view(np.uint8)[:, :8]
    return {
      "public_obs": self.public_obs[start:stop, env_id],
      "reward": self.reward[start:stop, env_id],
      "is_first": is_first,
      "is_last": done,
      "is_terminal": done,
      "action": self.action[start:stop, env_id],
      "consec": np.arange(self.config.sequence_length, dtype=np.int32),
      "stepid": stepid,
      "human_state_id": self.human_state_id[start:stop, env_id],
      "human_readiness": self.human_readiness[start:stop, env_id],
      "allostatic_load_total": self.allostatic_load_total[start:stop, env_id],
    }


def _require(arrays: dict[str, np.ndarray], key: str) -> np.ndarray:
  if key not in arrays:
    raise KeyError(f"dataset is missing required key: {key}")
  return arrays[key]
