"""Reward variants for the allostatic handover MVP."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class RewardVariant(str, Enum):
    TASK_ONLY = "task_only"
    ALLOSTATIC = "allostatic"
    SPEECH_PENALTY = "speech_penalty"

    @classmethod
    def from_name(cls, name: str | "RewardVariant") -> "RewardVariant":
        if isinstance(name, RewardVariant):
            return name
        return cls(str(name).strip().lower())


@dataclass
class RewardWeights:
    allostatic_load: float = 0.12
    forced_waiting: float = 0.04
    proxemic_stress: float = 0.08
    human_reach_effort: float = 0.06
    excessive_speech: float = 0.03
    uncomfortable_contact: float = 0.08
    speech_count: float = 0.03

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "RewardWeights":
        if values is None:
            return cls()
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in values.items() if key in allowed})


@dataclass
class RewardContext:
    allostatic_load_total: float = 0.0
    forced_waiting: float = 0.0
    proxemic_stress: float = 0.0
    human_reach_effort: float = 0.0
    robot_speech_count_step: float = 0.0
    uncomfortable_contact: float = 0.0


def compute_reward(
    task_reward: float,
    variant: RewardVariant | str,
    context: RewardContext,
    weights: RewardWeights | Mapping[str, Any] | None = None,
) -> float:
    """Apply an MVP reward variant to the base task reward."""
    variant = RewardVariant.from_name(variant)
    weights = weights if isinstance(weights, RewardWeights) else RewardWeights.from_mapping(weights)

    if variant == RewardVariant.TASK_ONLY:
        return float(task_reward)

    if variant == RewardVariant.SPEECH_PENALTY:
        return float(task_reward) - weights.speech_count * context.robot_speech_count_step

    penalty = (
        weights.allostatic_load * context.allostatic_load_total
        + weights.forced_waiting * context.forced_waiting
        + weights.proxemic_stress * context.proxemic_stress
        + weights.human_reach_effort * context.human_reach_effort
        + weights.excessive_speech * context.robot_speech_count_step
        + weights.uncomfortable_contact * context.uncomfortable_contact
    )
    return float(task_reward) - penalty
