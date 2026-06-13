"""Dependency-free handover environment for smoke tests and the dashboard."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from allostatic_handover.envs.allostatic_load import AllostaticLoadModel, InteractionFeatures
from allostatic_handover.envs.human_hidden_state import HumanHiddenStateMachine, HumanState
from allostatic_handover.envs.reward_variants import (
    RewardContext,
    RewardVariant,
    RewardWeights,
    compute_reward,
)
from allostatic_handover.envs.speech_events import (
    HumanSpeechEvent,
    RobotSpeechToken,
    robot_speech_from_scalar,
    speech_text,
)

try:
    from gymnasium import Env, spaces
except ImportError:
    Env = object
    spaces = None


@dataclass
class BoxSpace:
    low: np.ndarray
    high: np.ndarray
    shape: tuple[int, ...]

    def sample(self) -> np.ndarray:
        return np.random.uniform(self.low, self.high).astype(np.float32)


class MockAllostaticHandoverEnv(Env):
    """Small 2D handover simulator with the same allostatic interfaces.

    This is not the scientific simulator. It lets the repository's logging,
    plotting, training plumbing, and GUI run in a sandbox before MuJoCo and
    human-robot-gym are installed.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        reward_variant: str | RewardVariant = RewardVariant.TASK_ONLY,
        horizon: int = 160,
        privileged_observation: bool = False,
        speech_mode: str = "learned",
        seed: int | None = None,
        reward_weights: dict[str, Any] | None = None,
    ):
        self.reward_variant = RewardVariant.from_name(reward_variant)
        self.horizon = horizon
        self.privileged_observation = privileged_observation
        self.speech_mode = speech_mode
        self.reward_weights = RewardWeights.from_mapping(reward_weights)
        self.readiness_config = HumanReadinessConfig()
        action_low = np.array([-1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)
        action_high = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.action_space = (
            spaces.Box(low=action_low, high=action_high, dtype=np.float32)
            if spaces is not None
            else BoxSpace(low=action_low, high=action_high, shape=(5,))
        )
        obs_dim = 13 + (2 if privileged_observation else 0)
        obs_low = np.full(obs_dim, -np.inf, dtype=np.float32)
        obs_high = np.full(obs_dim, np.inf, dtype=np.float32)
        self.observation_space = (
            spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
            if spaces is not None
            else BoxSpace(low=obs_low, high=obs_high, shape=(obs_dim,))
        )
        self.rng = np.random.default_rng(seed)
        self.allostatic_load = AllostaticLoadModel(
            {
                "proximity_distance": 0.05,
                "reach_effort_cost": 0.08,
                "forced_waiting_cost": 0.04,
            }
        )
        self.human_fsm = HumanHiddenStateMachine(
            {
                "too_close_distance": 0.05,
                "overload_threshold": 4.5,
                "withdrawal_threshold": 8.0,
            }
        )
        self.reset(seed=seed)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.timestep = 0
        self.robot_pos = np.array([0.0, -0.25], dtype=np.float32)
        self.object_pos = np.array([0.42, -0.18], dtype=np.float32)
        self.human_hand_rest_pos = np.array([0.78, 0.48], dtype=np.float32)
        self.human_hand_extended_pos = np.array([0.95, 0.22], dtype=np.float32)
        self._human_reach_progress = 0.0
        self._human_reach_progress_sum = 0.0
        self.human_hand_pos = self.human_hand_rest_pos.copy()
        self.object_gripped = False
        self.success = False
        self._episode_return = 0.0
        self._robot_speech_count = 0
        self._silence_count = 0
        self._repeated_speech_count = 0
        self._human_waiting_time = 0.0
        self._human_reach_effort_sum = 0.0
        self._handover_time = None
        self._load_sum = 0.0
        self._load_max = 0.0
        self._load_area = 0.0
        self._raw_contact_this_step = False
        self._accepted_contact_this_step = False
        self._acceptance_blocked_this_step = False
        self._readiness_blocked_count = 0
        self._human_state_counts = {state.name: 0 for state in HumanState}
        self._last_robot_speech = RobotSpeechToken.SILENCE
        self._previous_robot_speech = RobotSpeechToken.SILENCE
        self._last_human_speech = HumanSpeechEvent.SILENCE
        self._last_transition_reason = "reset"
        self._human_readiness = self.readiness_config.initial
        self._readiness_belief = self.readiness_config.initial
        self._allostatic_load_proxy = 0.0
        self._readiness_sum = 0.0
        self._readiness_min = self._human_readiness
        self._readiness_max = self._human_readiness
        self._readiness_crossed_threshold_this_step = False
        self._readiness_hold_steps_remaining = 0
        self.allostatic_load.reset()
        self.human_fsm.reset()
        self.human_fsm.force_state(HumanState.DISTRACTED)
        obs = self._obs()
        return obs, self._info(0.0, 0.0)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.size < 5:
            action = np.pad(action, (0, 5 - action.size))
        speech = robot_speech_from_scalar(float(action[-1])) if self.speech_mode == "learned" else self._last_robot_speech
        self._last_robot_speech = speech
        if speech != RobotSpeechToken.SILENCE:
            self._robot_speech_count += 1
            if speech == self._previous_robot_speech:
                self._repeated_speech_count += 1
        else:
            self._silence_count += 1

        self.timestep += 1
        self._advance_readiness_from_speech()
        self._sync_fsm_with_readiness()
        self._advance_human_reach_animation()
        delta = np.clip(action[:2], -1.0, 1.0) * 0.035
        self.robot_pos = self.robot_pos + delta

        gripper_close = action[3] > 0.0
        if not self.object_gripped and gripper_close and np.linalg.norm(self.robot_pos - self.object_pos) < 0.08:
            self.object_gripped = True
        if self.object_gripped:
            self.object_pos = self.robot_pos.copy()

        distance_to_hand = float(np.linalg.norm(self.object_pos - self.human_hand_pos))
        raw_contact = self.object_gripped and distance_to_hand < 0.08
        self._raw_contact_this_step = bool(raw_contact)
        accepted_contact = raw_contact and self._readiness_allows_contact()
        self._accepted_contact_this_step = bool(accepted_contact)
        self._acceptance_blocked_this_step = bool(raw_contact and not accepted_contact)
        if accepted_contact:
            self.success = True
            self.object_gripped = False
        elif raw_contact:
            self._readiness_blocked_count += 1

        forced_waiting = 0.1 if self.object_gripped and distance_to_hand < 0.18 and not accepted_contact else 0.0
        features = InteractionFeatures(
            robot_speech=speech,
            previous_robot_speech=self._previous_robot_speech,
            forced_waiting=forced_waiting,
            proximity_distance=min(float(np.linalg.norm(self.robot_pos - self.human_hand_pos)), distance_to_hand),
            human_reach_effort=distance_to_hand,
            uncertainty=float(
                self.object_gripped
                and (
                    speech == RobotSpeechToken.SILENCE
                    or self._human_readiness < self.readiness_config.threshold
                )
            ),
            uncomfortable_contact=float(raw_contact and not accepted_contact),
            collision=0.0,
            smooth_progress=float(accepted_contact or self._readiness_crossed_threshold_this_step),
            step_seconds=0.1,
        )
        load_snapshot = self.allostatic_load.update(features)
        fsm_output = self.human_fsm.update(features, load_snapshot, contact=accepted_contact, success=self.success)
        self._last_human_speech = fsm_output.human_speech
        self._last_transition_reason = fsm_output.transition_reason
        self._sync_fsm_with_readiness()
        if self._readiness_crossed_threshold_this_step and self._last_human_speech == HumanSpeechEvent.SILENCE:
            self._last_human_speech = HumanSpeechEvent.READY
            self._last_transition_reason = "speech_cued_readiness"
        self._human_state_counts[self.human_fsm.state.name] += 1

        self._human_waiting_time += forced_waiting
        self._human_reach_effort_sum += distance_to_hand
        load_total = float(load_snapshot["allostatic_load_total"])
        self._load_sum += load_total
        self._load_max = max(self._load_max, load_total)
        self._load_area += load_total * 0.1
        self._update_load_proxy_from_features()
        self._record_readiness_metrics()
        if self.success and self._handover_time is None:
            self._handover_time = self.timestep * 0.1

        task_reward = 10.0 if self.success else -0.01
        if self.object_gripped:
            task_reward += 0.02
        context = RewardContext(
            allostatic_load_total=load_snapshot["allostatic_load_total"],
            forced_waiting=forced_waiting,
            proxemic_stress=load_snapshot["proxemic_stress"],
            human_reach_effort=distance_to_hand,
            robot_speech_count_step=1.0 if speech != RobotSpeechToken.SILENCE else 0.0,
            uncomfortable_contact=float(raw_contact and not accepted_contact),
        )
        reward = compute_reward(task_reward, self.reward_variant, context, self.reward_weights)
        self._episode_return += reward
        terminated = bool(self.success)
        truncated = self.timestep >= self.horizon
        obs = self._obs()
        info = self._info(task_reward, reward)
        if terminated or truncated:
            info["episode_metrics"] = self._episode_metrics()
        self._previous_robot_speech = self._last_robot_speech
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        return None

    def _obs(self) -> np.ndarray:
        values = [
            *self.robot_pos.tolist(),
            *self.object_pos.tolist(),
            *self.human_hand_pos.tolist(),
            float(self.object_gripped),
            float(self._last_human_speech.value),
            float(self._previous_robot_speech.value),
            float(self._readiness_belief),
            float(self._allostatic_load_proxy),
            float(self.timestep / max(1, self.horizon)),
            float(self.success),
        ]
        if self.privileged_observation:
            values.extend([float(self.human_fsm.state.value), float(self.allostatic_load.total)])
        return np.asarray(values, dtype=np.float32)

    def _info(self, task_reward: float, reward: float) -> dict[str, Any]:
        load = self.allostatic_load.snapshot()
        conversation = []
        if self._last_robot_speech != RobotSpeechToken.SILENCE:
            conversation.append(
                {
                    "speaker": "robot",
                    "token": self._last_robot_speech.name.lower(),
                    "text": speech_text(self._last_robot_speech),
                }
            )
        if self._last_human_speech != HumanSpeechEvent.SILENCE:
            conversation.append(
                {
                    "speaker": "human",
                    "token": self._last_human_speech.name.lower(),
                    "text": speech_text(self._last_human_speech),
                }
            )
        return {
            **load,
            **self.allostatic_load.last_delta,
            "reward_variant": self.reward_variant.value,
            "base_task_reward": float(task_reward),
            "allostatic_reward": float(reward),
            "success": bool(self.success),
            "human_state": self.human_fsm.state.name,
            "human_state_id": int(self.human_fsm.state.value),
            "human_speech_event": self._last_human_speech.name.lower(),
            "human_speech_text": speech_text(self._last_human_speech),
            "robot_speech": self._last_robot_speech.name.lower(),
            "robot_speech_text": speech_text(self._last_robot_speech),
            "conversation": conversation,
            "transition_reason": self._last_transition_reason,
            "robot_speech_count": self._robot_speech_count,
            "silence_count": self._silence_count,
            "silence_ratio": self._ratio(self._silence_count),
            "repeated_speech_count": self._repeated_speech_count,
            "human_waiting_time": self._human_waiting_time,
            "human_reach_effort": float(np.linalg.norm(self.object_pos - self.human_hand_pos)),
            "human_reach_effort_sum": self._human_reach_effort_sum,
            "allostatic_load_mean": self._mean_load(),
            "allostatic_load_max": self._load_max,
            "allostatic_load_final": load["allostatic_load_total"],
            "allostatic_load_area": self._load_area,
            "withdrawal_count": self.human_fsm.withdrawal_count,
            "overload_count": self.human_fsm.overload_count,
            "handover_time": self._handover_time,
            "collision_count": 0,
            "drop_count": 0,
            "object_in_human_hand": bool(self._accepted_contact_this_step or self.success),
            "acceptance_blocked": self._acceptance_blocked_this_step,
            "raw_handover_contact": self._raw_contact_this_step,
            "accepted_handover_contact": self._accepted_contact_this_step,
            "human_readiness": self._human_readiness,
            "human_readiness_belief": self._readiness_belief,
            "human_readiness_mean": self._mean_readiness(),
            "human_readiness_min": self._readiness_min,
            "human_readiness_max": self._readiness_max,
            "human_readiness_final": self._human_readiness,
            "readiness_threshold": self.readiness_config.threshold,
            "readiness_blocked_count": self._readiness_blocked_count,
            "readiness_hold_steps_remaining": self._readiness_hold_steps_remaining,
            "allostatic_load_proxy": self._allostatic_load_proxy,
            "human_reach_progress": self._human_reach_progress,
            "human_reach_progress_mean": self._mean_reach_progress(),
            "animation_gated_by_readiness": self._human_readiness < self.readiness_config.threshold,
            "reach_out_started_count": int(self._human_reach_progress > 0.0),
            "robot_eef_pos": [float(self.robot_pos[0]), float(self.robot_pos[1]), 0.0],
            "object_pos": [float(self.object_pos[0]), float(self.object_pos[1]), 0.0],
            "human_hand_pos": [float(self.human_hand_pos[0]), float(self.human_hand_pos[1]), 0.0],
            "collision": False,
            "n_collisions": 0,
            **self._human_state_ratios(),
        }

    def _episode_metrics(self) -> dict[str, Any]:
        metrics = {
            "success": float(self.success),
            "return": self._episode_return,
            "length": self.timestep,
            "handover_time": self._handover_time,
            "robot_speech_count": self._robot_speech_count,
            "silence_count": self._silence_count,
            "silence_ratio": self._ratio(self._silence_count),
            "repeated_speech_count": self._repeated_speech_count,
            "human_waiting_time": self._human_waiting_time,
            "human_reach_effort": self._human_reach_effort_sum,
            "allostatic_load_mean": self._mean_load(),
            "allostatic_load_max": self._load_max,
            "allostatic_load_final": self.allostatic_load.total,
            "allostatic_load_area": self._load_area,
            "withdrawal_count": self.human_fsm.withdrawal_count,
            "overload_count": self.human_fsm.overload_count,
            "collision_count": 0,
            "drop_count": 0,
            "object_in_human_hand": float(self.success),
            "human_readiness": self._human_readiness,
            "human_readiness_belief": self._readiness_belief,
            "human_readiness_mean": self._mean_readiness(),
            "human_readiness_min": self._readiness_min,
            "human_readiness_max": self._readiness_max,
            "human_readiness_final": self._human_readiness,
            "readiness_threshold": self.readiness_config.threshold,
            "readiness_blocked_count": self._readiness_blocked_count,
            "readiness_hold_steps_remaining": self._readiness_hold_steps_remaining,
            "allostatic_load_proxy": self._allostatic_load_proxy,
            "human_reach_progress": self._human_reach_progress,
            "human_reach_progress_mean": self._mean_reach_progress(),
            "animation_gated_by_readiness": self._human_readiness < self.readiness_config.threshold,
            "reach_out_started_count": int(self._human_reach_progress > 0.0),
            **self.allostatic_load.snapshot(),
        }
        metrics.update(self._human_state_ratios())
        return metrics

    def _mean_load(self) -> float:
        return self._load_sum / max(1, self.timestep)

    def _mean_readiness(self) -> float:
        return self._readiness_sum / max(1, self.timestep)

    def _mean_reach_progress(self) -> float:
        return self._human_reach_progress_sum / max(1, self.timestep)

    def _ratio(self, count: int) -> float:
        return float(count) / max(1, self.timestep)

    def _human_state_ratios(self) -> dict[str, float]:
        return {
            f"human_state_{state.name.lower()}_ratio": self._ratio(self._human_state_counts.get(state.name, 0))
            for state in HumanState
        }

    def _advance_readiness_from_speech(self) -> None:
        cfg = self.readiness_config
        repeated = (
            self._last_robot_speech != RobotSpeechToken.SILENCE
            and self._last_robot_speech == self._previous_robot_speech
        )
        before = self._human_readiness
        effect = cfg.speech_effects.get(self._last_robot_speech, 0.0)
        if repeated:
            effect *= cfg.repeated_effect_scale
        if self._last_robot_speech in cfg.readiness_hold_tokens and effect > 0.0:
            self._readiness_hold_steps_remaining = max(
                self._readiness_hold_steps_remaining,
                cfg.readiness_hold_steps,
            )
        hold_active = self._readiness_hold_steps_remaining > 0
        load_drag = min(
            cfg.max_load_drag,
            cfg.load_drag * max(0.0, self.allostatic_load.total - cfg.load_resilience_threshold),
        )
        decay = 0.0 if hold_active else cfg.decay
        self._human_readiness = self._clip_readiness(self._human_readiness + effect - decay - load_drag)
        if hold_active:
            self._human_readiness = max(self._human_readiness, cfg.readiness_hold_floor)
            self._readiness_hold_steps_remaining -= 1

        belief_decay = 0.0 if hold_active else cfg.belief_decay
        belief_delta = effect - belief_decay
        if self._last_human_speech in {HumanSpeechEvent.READY, HumanSpeechEvent.GOT_IT}:
            belief_delta += cfg.human_ready_belief_bonus
        elif self._last_human_speech in {HumanSpeechEvent.WAIT, HumanSpeechEvent.CONFUSED, HumanSpeechEvent.TOO_CLOSE}:
            belief_delta -= cfg.human_not_ready_belief_penalty
        self._readiness_belief = self._clip_readiness(self._readiness_belief + belief_delta)
        self._readiness_crossed_threshold_this_step = before < cfg.threshold <= self._human_readiness

    def _sync_fsm_with_readiness(self) -> None:
        cfg = self.readiness_config
        if self.human_fsm.state in {HumanState.WITHDRAWING, HumanState.GRASPING}:
            return
        if self._human_readiness >= cfg.threshold and self.allostatic_load.total < cfg.withdrawal_load_threshold:
            if self.human_fsm.state in {HumanState.DISTRACTED, HumanState.HESITANT, HumanState.OVERLOADED}:
                self.human_fsm.force_state(HumanState.READY)
        elif self._human_readiness < cfg.threshold and self.human_fsm.state == HumanState.READY:
            self.human_fsm.force_state(HumanState.DISTRACTED)

    def _readiness_allows_contact(self) -> bool:
        cfg = self.readiness_config
        if self.human_fsm.state == HumanState.WITHDRAWING:
            return False
        if self._human_reach_progress < cfg.min_reach_progress_for_contact:
            return False
        return self._human_readiness >= cfg.threshold

    def _advance_human_reach_animation(self) -> None:
        cfg = self.readiness_config
        if self._human_readiness >= cfg.threshold and self.human_fsm.state != HumanState.WITHDRAWING:
            self._human_reach_progress = min(1.0, self._human_reach_progress + cfg.mock_reach_extend_rate)
        else:
            self._human_reach_progress = max(0.0, self._human_reach_progress - cfg.mock_reach_retract_rate)
        self.human_hand_pos = (
            (1.0 - self._human_reach_progress) * self.human_hand_rest_pos
            + self._human_reach_progress * self.human_hand_extended_pos
        ).astype(np.float32)

    def _update_load_proxy_from_features(self) -> None:
        delta = self.allostatic_load.last_delta
        increment = (
            float(delta.get("speech_cost", 0.0))
            + float(delta.get("repeated_speech_cost", 0.0))
            + float(delta.get("forced_waiting_cost", 0.0))
            + float(delta.get("contact_discomfort_cost", 0.0))
            + float(delta.get("uncertainty_cost", 0.0))
            + 0.5 * float(delta.get("proximity_cost", 0.0))
        )
        cfg = self.allostatic_load.config
        self._allostatic_load_proxy = max(
            0.0,
            min(cfg.max_load, cfg.rho * self._allostatic_load_proxy + increment - cfg.recovery),
        )

    def _record_readiness_metrics(self) -> None:
        self._readiness_sum += self._human_readiness
        self._readiness_min = min(self._readiness_min, self._human_readiness)
        self._readiness_max = max(self._readiness_max, self._human_readiness)
        self._human_reach_progress_sum += self._human_reach_progress

    @staticmethod
    def _clip_readiness(value: float) -> float:
        return max(0.0, min(1.0, float(value)))


@dataclass
class HumanReadinessConfig:
    initial: float = 0.20
    threshold: float = 0.55
    decay: float = 0.004
    belief_decay: float = 0.003
    repeated_effect_scale: float = 0.65
    load_resilience_threshold: float = 7.0
    withdrawal_load_threshold: float = 9.0
    load_drag: float = 0.01
    max_load_drag: float = 0.04
    min_reach_progress_for_contact: float = 0.65
    readiness_hold_steps: int = 160
    readiness_hold_floor: float = 0.72
    readiness_hold_tokens: tuple[RobotSpeechToken, ...] = field(
        default_factory=lambda: (
            RobotSpeechToken.ANNOUNCE_HANDOVER,
            RobotSpeechToken.SAY_RELEASING,
        )
    )
    mock_reach_extend_rate: float = 0.08
    mock_reach_retract_rate: float = 0.04
    human_ready_belief_bonus: float = 0.10
    human_not_ready_belief_penalty: float = 0.12
    speech_effects: dict[RobotSpeechToken, float] = field(
        default_factory=lambda: {
            RobotSpeechToken.SILENCE: 0.0,
            RobotSpeechToken.ANNOUNCE_HANDOVER: 0.45,
            RobotSpeechToken.ASK_READY: 0.38,
            RobotSpeechToken.REASSURE: 0.24,
            RobotSpeechToken.SAY_WAITING: 0.18,
            RobotSpeechToken.SAY_RELEASING: 0.45,
            RobotSpeechToken.ASK_CONFIRMATION: 0.05,
        }
    )
