"""Human hidden-state finite state machine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Mapping

from allostatic_handover.envs.allostatic_load import InteractionFeatures
from allostatic_handover.envs.speech_events import HumanSpeechEvent, RobotSpeechToken


class HumanState(IntEnum):
    READY = 0
    HESITANT = 1
    DISTRACTED = 2
    OVERLOADED = 3
    WITHDRAWING = 4
    GRASPING = 5


@dataclass
class HumanFSMConfig:
    overload_threshold: float = 7.0
    withdrawal_threshold: float = 9.0
    too_close_distance: float = 0.16
    repeated_ask_ready_limit: int = 3
    distracted_every_n_steps: int = 0
    hesitation_recovery_load: float = 2.5
    grasp_contact_threshold: float = 0.5

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "HumanFSMConfig":
        if values is None:
            return cls()
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in values.items() if key in allowed})


@dataclass
class HumanFSMOutput:
    previous_state: HumanState
    state: HumanState
    human_speech: HumanSpeechEvent
    transition_reason: str


class HumanHiddenStateMachine:
    """FSM for the human latent state z_t^H."""

    def __init__(self, config: HumanFSMConfig | Mapping[str, Any] | None = None):
        self.config = config if isinstance(config, HumanFSMConfig) else HumanFSMConfig.from_mapping(config)
        self.state = HumanState.READY
        self.step_count = 0
        self.repeated_ask_ready_count = 0
        self.withdrawal_count = 0
        self.overload_count = 0

    def reset(self) -> None:
        self.state = HumanState.READY
        self.step_count = 0
        self.repeated_ask_ready_count = 0
        self.withdrawal_count = 0
        self.overload_count = 0

    def force_state(self, state: HumanState) -> None:
        self.state = HumanState(state)

    def accepts_handover(self) -> bool:
        return self.state in {HumanState.READY, HumanState.GRASPING}

    def update(
        self,
        features: InteractionFeatures | Mapping[str, Any],
        load_snapshot: Mapping[str, float],
        contact: bool = False,
        success: bool = False,
    ) -> HumanFSMOutput:
        if isinstance(features, Mapping):
            features = InteractionFeatures(**features)

        self.step_count += 1
        previous = self.state
        cfg = self.config
        load_total = float(load_snapshot.get("allostatic_load_total", 0.0))
        speech = features.robot_speech
        repeated_speech = speech != RobotSpeechToken.SILENCE and speech == features.previous_robot_speech

        if speech == RobotSpeechToken.ASK_READY and repeated_speech:
            self.repeated_ask_ready_count += 1
        elif speech != RobotSpeechToken.ASK_READY:
            self.repeated_ask_ready_count = 0

        human_speech = HumanSpeechEvent.SILENCE
        reason = "stable"
        next_state = self.state

        too_close = (
            features.proximity_distance is not None
            and features.proximity_distance < cfg.too_close_distance
        )

        if success:
            next_state = HumanState.READY
            human_speech = HumanSpeechEvent.GOT_IT
            reason = "handover_success"
        elif load_total >= cfg.withdrawal_threshold or self.state == HumanState.WITHDRAWING:
            next_state = HumanState.WITHDRAWING
            human_speech = HumanSpeechEvent.WAIT
            reason = "load_withdrawal"
        elif load_total >= cfg.overload_threshold or self.repeated_ask_ready_count >= cfg.repeated_ask_ready_limit:
            next_state = HumanState.OVERLOADED
            human_speech = HumanSpeechEvent.CONFUSED
            reason = "overload"
        elif too_close:
            next_state = HumanState.HESITANT
            human_speech = HumanSpeechEvent.TOO_CLOSE
            reason = "too_close"
        elif contact and self.state in {HumanState.READY, HumanState.GRASPING}:
            next_state = HumanState.GRASPING
            human_speech = HumanSpeechEvent.READY
            reason = "stable_contact"
        elif self.state == HumanState.HESITANT and speech == RobotSpeechToken.REASSURE and load_total < cfg.hesitation_recovery_load:
            next_state = HumanState.READY
            human_speech = HumanSpeechEvent.READY
            reason = "reassured"
        elif self.state == HumanState.OVERLOADED and speech in {RobotSpeechToken.REASSURE, RobotSpeechToken.SAY_WAITING}:
            next_state = HumanState.HESITANT
            human_speech = HumanSpeechEvent.WAIT
            reason = "demand_reduced"
        elif cfg.distracted_every_n_steps > 0 and self.step_count % cfg.distracted_every_n_steps == 0:
            next_state = HumanState.DISTRACTED
            human_speech = HumanSpeechEvent.CONFUSED
            reason = "exogenous_distraction"
        elif self.state == HumanState.DISTRACTED and speech in {
            RobotSpeechToken.ANNOUNCE_HANDOVER,
            RobotSpeechToken.REASSURE,
        }:
            next_state = HumanState.READY
            human_speech = HumanSpeechEvent.READY
            reason = "attention_recovered"
        elif self.state in {HumanState.OVERLOADED, HumanState.HESITANT} and load_total < cfg.hesitation_recovery_load:
            next_state = HumanState.READY
            human_speech = HumanSpeechEvent.READY
            reason = "load_recovered"

        self.state = next_state
        if self.state == HumanState.OVERLOADED and previous != HumanState.OVERLOADED:
            self.overload_count += 1
        if self.state == HumanState.WITHDRAWING and previous != HumanState.WITHDRAWING:
            self.withdrawal_count += 1

        return HumanFSMOutput(
            previous_state=previous,
            state=self.state,
            human_speech=human_speech,
            transition_reason=reason,
        )
