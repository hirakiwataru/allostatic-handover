"""Local JSONL/CSV logging for experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


EPISODE_FIELDS = [
    "episode_id",
    "reward_variant",
    "policy",
    "success",
    "goal_reached",
    "return",
    "length",
    "handover_time",
    "robot_speech_count",
    "silence_count",
    "silence_ratio",
    "repeated_speech_count",
    "human_waiting_time",
    "human_reach_effort",
    "human_readiness",
    "human_readiness_belief",
    "human_readiness_mean",
    "human_readiness_min",
    "human_readiness_max",
    "human_readiness_final",
    "readiness_threshold",
    "readiness_blocked_count",
    "readiness_hold_steps_remaining",
    "human_reach_progress",
    "human_reach_progress_mean",
    "animation_gated_by_readiness",
    "reach_out_started_count",
    "allostatic_load_proxy",
    "allostatic_load_total",
    "allostatic_load_mean",
    "allostatic_load_max",
    "allostatic_load_final",
    "allostatic_load_area",
    "attention_load",
    "turn_taking_load",
    "proxemic_stress",
    "motor_adaptation_cost",
    "annoyance",
    "trust",
    "human_state_ready_ratio",
    "human_state_hesitant_ratio",
    "human_state_distracted_ratio",
    "human_state_overloaded_ratio",
    "human_state_withdrawing_ratio",
    "human_state_grasping_ratio",
    "withdrawal_count",
    "overload_count",
    "collision_count",
    "drop_count",
    "object_in_human_hand",
]


class EpisodeLogger:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.steps_path = self.output_dir / "steps.jsonl"
        self.episodes_jsonl_path = self.output_dir / "episodes.jsonl"
        self.episodes_csv_path = self.output_dir / "episodes.csv"
        self._csv_initialized = self.episodes_csv_path.exists() and self.episodes_csv_path.stat().st_size > 0

    def log_step(
        self,
        episode_id: int,
        step_id: int,
        obs: Any,
        action: Any,
        reward: float,
        done: bool,
        info: Mapping[str, Any],
    ) -> None:
        row = {
            "episode_id": episode_id,
            "step": step_id,
            "reward": float(reward),
            "done": bool(done),
            "obs": _jsonable(obs),
            "action": _jsonable(action),
            "info": _jsonable(dict(info)),
        }
        with self.steps_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def log_episode(self, metrics: Mapping[str, Any]) -> None:
        clean = _jsonable(dict(metrics))
        with self.episodes_jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")

        row = {field: clean.get(field, "") for field in EPISODE_FIELDS}
        with self.episodes_csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EPISODE_FIELDS)
            if not self._csv_initialized:
                writer.writeheader()
                self._csv_initialized = True
            writer.writerow(row)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "name") and hasattr(value, "value"):
        return getattr(value, "name")
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)
