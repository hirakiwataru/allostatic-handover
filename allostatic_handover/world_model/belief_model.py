"""Frozen PyTorch belief estimator distilled from world-model rollouts.

The project keeps the DreamerV3 repository as an external dependency and saves
Dreamer-style world-model artifacts, but PPO runtime uses this compact PyTorch
recurrent estimator so Mjlab observations can be computed on the same device as
the policy without crossing into JAX.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_DREAMERV3_PATH = "/mnt/k_iwamoto/sim_data/Projects/dreamerv3"
DEFAULT_BELIEF_MODEL_PATH = (
  "/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/"
  "outputs/world_model/latest/belief_distill.pt"
)


@dataclass
class BeliefModelConfig:
  """Architecture and input dimensions for the recurrent belief estimator."""

  public_obs_dim: int = 31
  action_dim: int = 5
  hidden_dim: int = 96
  belief_dim: int = 16
  num_human_states: int = 6


class BeliefEstimator(nn.Module):
  """GRU belief model with auxiliary heads for hidden human-state labels."""

  def __init__(
    self,
    config: BeliefModelConfig,
    normalization: dict[str, list[float]] | None = None,
  ) -> None:
    super().__init__()
    self.config = config
    input_dim = config.public_obs_dim + config.action_dim
    self.gru = nn.GRUCell(input_dim, config.hidden_dim)
    self.belief_head = nn.Sequential(
      nn.LayerNorm(config.hidden_dim),
      nn.Linear(config.hidden_dim, config.belief_dim),
      nn.Tanh(),
    )
    self.state_head = nn.Linear(config.hidden_dim, config.num_human_states)
    self.readiness_head = nn.Sequential(nn.Linear(config.hidden_dim, 1), nn.Sigmoid())
    self.load_head = nn.Sequential(nn.Linear(config.hidden_dim, 1), nn.Softplus())

    self.register_buffer("obs_mean", torch.zeros(config.public_obs_dim))
    self.register_buffer("obs_std", torch.ones(config.public_obs_dim))
    self.register_buffer("action_mean", torch.zeros(config.action_dim))
    self.register_buffer("action_std", torch.ones(config.action_dim))
    if normalization is not None:
      self.set_normalization(normalization)

  def set_normalization(self, normalization: dict[str, list[float]]) -> None:
    """Load mean/std statistics into buffers."""
    device = self.obs_mean.device
    dtype = self.obs_mean.dtype
    self.obs_mean.copy_(torch.as_tensor(normalization["public_obs_mean"], device=device, dtype=dtype))
    self.obs_std.copy_(torch.as_tensor(normalization["public_obs_std"], device=device, dtype=dtype))
    self.action_mean.copy_(torch.as_tensor(normalization["action_mean"], device=device, dtype=dtype))
    self.action_std.copy_(torch.as_tensor(normalization["action_std"], device=device, dtype=dtype))

  def initial_state(self, batch_size: int, device: torch.device | str | None = None) -> torch.Tensor:
    """Return a zero recurrent hidden state."""
    param = next(self.parameters())
    return torch.zeros(
      batch_size,
      self.config.hidden_dim,
      device=device or param.device,
      dtype=param.dtype,
    )

  def _normalized_input(self, public_obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    obs = (public_obs - self.obs_mean) / self.obs_std.clamp_min(1e-6)
    act = (action - self.action_mean) / self.action_std.clamp_min(1e-6)
    return torch.cat([obs, act], dim=-1)

  def step(
    self,
    public_obs: torch.Tensor,
    action: torch.Tensor,
    hidden: torch.Tensor | None = None,
    reset_mask: torch.Tensor | None = None,
  ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Advance the recurrent belief by one step."""
    if hidden is None:
      hidden = self.initial_state(public_obs.shape[0], public_obs.device)
    if reset_mask is not None:
      keep = (~reset_mask.bool()).float().unsqueeze(-1)
      hidden = hidden * keep
    hidden = self.gru(self._normalized_input(public_obs, action), hidden)
    return self._heads(hidden), hidden

  def forward_sequence(
    self,
    public_obs: torch.Tensor,
    action: torch.Tensor,
    done: torch.Tensor | None = None,
  ) -> dict[str, torch.Tensor]:
    """Run a batch of sequences.

    Args:
      public_obs: ``[batch, time, public_obs_dim]`` tensor.
      action: ``[batch, time, action_dim]`` tensor.
      done: optional ``[batch, time]`` boolean/float tensor. A true value resets
        the hidden state before processing that timestep.
    """
    batch, time, _ = public_obs.shape
    hidden = self.initial_state(batch, public_obs.device)
    outputs: dict[str, list[torch.Tensor]] = {
      "belief": [],
      "human_state_logits": [],
      "human_state_probs": [],
      "readiness": [],
      "load": [],
    }
    for index in range(time):
      reset_mask = None if done is None else done[:, index].bool()
      step_out, hidden = self.step(public_obs[:, index], action[:, index], hidden, reset_mask)
      for key, value in step_out.items():
        outputs[key].append(value)
    return {key: torch.stack(values, dim=1) for key, values in outputs.items()}

  def _heads(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
    logits = self.state_head(hidden)
    return {
      "belief": self.belief_head(hidden),
      "human_state_logits": logits,
      "human_state_probs": F.softmax(logits, dim=-1),
      "readiness": self.readiness_head(hidden),
      "load": self.load_head(hidden),
    }


def save_belief_model(
  path: str | Path,
  model: BeliefEstimator,
  normalization: dict[str, list[float]],
  metrics: dict[str, float] | None = None,
  dreamerv3_path: str = DEFAULT_DREAMERV3_PATH,
  extra: dict[str, Any] | None = None,
) -> None:
  """Save a portable belief model checkpoint."""
  payload = {
    "format": "allostatic_handover_belief_distill_v1",
    "config": asdict(model.config),
    "state_dict": model.state_dict(),
    "normalization": normalization,
    "metrics": metrics or {},
    "dreamerv3_path": dreamerv3_path,
    "extra": extra or {},
  }
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  torch.save(payload, path)


def load_belief_model(
  path: str | Path,
  map_location: torch.device | str | None = None,
) -> tuple[BeliefEstimator, dict[str, Any]]:
  """Load a saved belief estimator and return ``(model, metadata)``."""
  payload = torch.load(path, map_location=map_location or "cpu", weights_only=False)
  config = BeliefModelConfig(**payload["config"])
  model = BeliefEstimator(config, payload.get("normalization"))
  model.load_state_dict(payload["state_dict"])
  model.eval()
  return model, payload
