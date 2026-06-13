"""Allostatic load model for the handover MVP."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Mapping

from allostatic_handover.envs.speech_events import RobotSpeechToken


@dataclass
class AllostaticLoadConfig:
    rho: float = 0.94
    max_load: float = 10.0
    recovery: float = 0.015
    helpful_cue_bonus: float = 0.08
    speech_cost: float = 0.035
    repeated_speech_cost: float = 0.11
    ask_ready_cost: float = 0.06
    forced_waiting_cost: float = 0.055
    proximity_distance: float = 0.22
    proximity_cost: float = 0.42
    reach_threshold: float = 0.18
    reach_effort_cost: float = 0.55
    uncertainty_cost: float = 0.05
    contact_discomfort_cost: float = 0.30
    trust_recovery: float = 0.012
    trust_loss_repeated_speech: float = 0.02
    trust_loss_collision: float = 0.08
    trust_loss_too_close: float = 0.035
    trust_bonus_helpful_cue: float = 0.025
    trust_deficit_weight: float = 1.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "AllostaticLoadConfig":
        if values is None:
            return cls()
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in values.items() if key in allowed})


@dataclass
class LoadComponents:
    attention_load: float = 0.0
    turn_taking_load: float = 0.0
    proxemic_stress: float = 0.0
    motor_adaptation_cost: float = 0.0
    annoyance: float = 0.0
    trust: float = 0.75

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class InteractionFeatures:
    robot_speech: RobotSpeechToken = RobotSpeechToken.SILENCE
    previous_robot_speech: RobotSpeechToken = RobotSpeechToken.SILENCE
    forced_waiting: float = 0.0
    proximity_distance: float | None = None
    human_reach_effort: float = 0.0
    uncertainty: float = 0.0
    uncomfortable_contact: float = 0.0
    collision: float = 0.0
    smooth_progress: float = 0.0
    step_seconds: float = 0.1


class AllostaticLoadModel:
    """Stateful allostatic load accumulator.

    The model is deliberately simple and interpretable. It is not intended as a
    physiological model; it encodes the interaction costs needed to test the MVP
    hypothesis.
    """

    def __init__(self, config: AllostaticLoadConfig | Mapping[str, Any] | None = None):
        self.config = config if isinstance(config, AllostaticLoadConfig) else AllostaticLoadConfig.from_mapping(config)
        self.components = LoadComponents()
        self.last_delta: dict[str, float] = {}

    def reset(self) -> None:
        self.components = LoadComponents()
        self.last_delta = {}

    @property
    def total(self) -> float:
        c = self.components
        trust_deficit = max(0.0, 1.0 - c.trust) * self.config.trust_deficit_weight
        total = (
            c.attention_load
            + c.turn_taking_load
            + c.proxemic_stress
            + c.motor_adaptation_cost
            + c.annoyance
            + trust_deficit
        )
        return max(0.0, min(self.config.max_load, total))

    def snapshot(self) -> dict[str, float]:
        values = self.components.as_dict()
        values["allostatic_load_total"] = self.total
        return values

    def update(self, features: InteractionFeatures | Mapping[str, Any]) -> dict[str, float]:
        if isinstance(features, Mapping):
            features = InteractionFeatures(**features)

        cfg = self.config
        c = self.components
        seconds_scale = max(0.0, features.step_seconds / 0.1)

        speech_cost = 0.0
        repeated_speech_cost = 0.0
        ask_ready_cost = 0.0
        helpful_cue = 0.0

        if features.robot_speech != RobotSpeechToken.SILENCE:
            speech_cost = cfg.speech_cost
            if features.robot_speech == features.previous_robot_speech:
                repeated_speech_cost = cfg.repeated_speech_cost
            if features.robot_speech == RobotSpeechToken.ASK_READY:
                ask_ready_cost = cfg.ask_ready_cost
            if features.robot_speech in {
                RobotSpeechToken.REASSURE,
                RobotSpeechToken.ANNOUNCE_HANDOVER,
                RobotSpeechToken.SAY_RELEASING,
            }:
                helpful_cue = cfg.helpful_cue_bonus

        proximity_cost = 0.0
        if features.proximity_distance is not None:
            margin = max(0.0, cfg.proximity_distance - features.proximity_distance)
            proximity_cost = cfg.proximity_cost * margin / max(cfg.proximity_distance, 1e-6)

        reach_effort_cost = cfg.reach_effort_cost * max(0.0, features.human_reach_effort - cfg.reach_threshold)
        forced_waiting_cost = cfg.forced_waiting_cost * max(0.0, features.forced_waiting)
        uncertainty_cost = cfg.uncertainty_cost * max(0.0, features.uncertainty)
        contact_cost = cfg.contact_discomfort_cost * max(0.0, features.uncomfortable_contact)

        c.attention_load = self._clip_load(
            cfg.rho * c.attention_load + speech_cost + uncertainty_cost - cfg.recovery * seconds_scale
        )
        c.turn_taking_load = self._clip_load(
            cfg.rho * c.turn_taking_load
            + ask_ready_cost
            + repeated_speech_cost
            + forced_waiting_cost
            - cfg.recovery * seconds_scale
        )
        c.proxemic_stress = self._clip_load(
            cfg.rho * c.proxemic_stress + proximity_cost + contact_cost - cfg.recovery * seconds_scale
        )
        c.motor_adaptation_cost = self._clip_load(
            cfg.rho * c.motor_adaptation_cost + reach_effort_cost - cfg.recovery * seconds_scale
        )
        c.annoyance = self._clip_load(
            cfg.rho * c.annoyance
            + repeated_speech_cost
            + ask_ready_cost
            + 0.5 * forced_waiting_cost
            - helpful_cue
            - cfg.recovery * seconds_scale
        )

        trust_delta = (
            cfg.trust_recovery * seconds_scale
            + cfg.trust_bonus_helpful_cue * (1.0 if helpful_cue > 0.0 else 0.0)
            - cfg.trust_loss_repeated_speech * (1.0 if repeated_speech_cost > 0.0 else 0.0)
            - cfg.trust_loss_collision * max(0.0, features.collision)
            - cfg.trust_loss_too_close * (1.0 if proximity_cost > 0.0 else 0.0)
        )
        c.trust = max(0.0, min(1.0, c.trust + trust_delta))

        if features.smooth_progress > 0.0:
            self.apply_recovery(cfg.helpful_cue_bonus * 0.5 * features.smooth_progress)

        self.last_delta = {
            "speech_cost": speech_cost,
            "repeated_speech_cost": repeated_speech_cost,
            "forced_waiting_cost": forced_waiting_cost,
            "proximity_cost": proximity_cost,
            "reach_effort_cost": reach_effort_cost,
            "uncertainty_cost": uncertainty_cost,
            "contact_discomfort_cost": contact_cost,
            "helpful_cue_bonus": helpful_cue,
        }
        return self.snapshot()

    def apply_recovery(self, amount: float) -> None:
        amount = max(0.0, amount)
        c = self.components
        c.attention_load = self._clip_load(c.attention_load - amount)
        c.turn_taking_load = self._clip_load(c.turn_taking_load - amount)
        c.proxemic_stress = self._clip_load(c.proxemic_stress - amount)
        c.motor_adaptation_cost = self._clip_load(c.motor_adaptation_cost - amount)
        c.annoyance = self._clip_load(c.annoyance - amount)

    def _clip_load(self, value: float) -> float:
        return max(0.0, min(self.config.max_load, float(value)))
