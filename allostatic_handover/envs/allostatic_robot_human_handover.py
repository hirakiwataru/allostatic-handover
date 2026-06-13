"""Allostatic extension of human-robot-gym RobotHumanHandoverCart."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

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
    from robosuite.utils.observables import Observable, sensor
    from human_robot_gym.environments.manipulation.robot_human_handover_cartesian_env import (
        RobotHumanHandoverCart,
        RobotHumanHandoverPhase,
    )
    from human_robot_gym.utils.animation_utils import layered_sin_modulations

    _HAS_HRGYM = True
except ImportError:
    _HAS_HRGYM = False
    Observable = None
    sensor = None
    RobotHumanHandoverPhase = None
    layered_sin_modulations = None

    class RobotHumanHandoverCart:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "human_robot_gym and robosuite are required for AllostaticRobotHumanHandoverCart. "
                "Use backend='mock' for dependency-free smoke tests."
            )


class AllostaticRobotHumanHandoverCart(RobotHumanHandoverCart):
    """Robot-to-human handover env with speech, hidden state, and load model."""

    def __init__(
        self,
        *args,
        reward_variant: str | RewardVariant = RewardVariant.TASK_ONLY,
        speech_mode: str = "learned",
        privileged_observation: bool = False,
        allostatic_config: Mapping[str, Any] | None = None,
        human_fsm_config: Mapping[str, Any] | None = None,
        reward_weights: Mapping[str, Any] | None = None,
        acceptance_delay_steps: int = 8,
        **kwargs,
    ):
        self.reward_variant = RewardVariant.from_name(reward_variant)
        self.speech_mode = speech_mode
        self.privileged_observation = privileged_observation
        self.allostatic_load = AllostaticLoadModel(allostatic_config)
        self.human_fsm = HumanHiddenStateMachine(human_fsm_config)
        self.reward_weights = RewardWeights.from_mapping(reward_weights)
        self.acceptance_delay_steps = acceptance_delay_steps

        self._last_robot_speech = RobotSpeechToken.SILENCE
        self._previous_robot_speech = RobotSpeechToken.SILENCE
        self._last_human_speech = HumanSpeechEvent.SILENCE
        self._robot_speech_count = 0
        self._silence_count = 0
        self._repeated_speech_count = 0
        self._human_waiting_time = 0.0
        self._human_reach_effort_sum = 0.0
        self._handover_time = None
        self._episode_return = 0.0
        self._episode_step_count = 0
        self._load_sum = 0.0
        self._load_max = 0.0
        self._load_area = 0.0
        self._collision_count = 0
        self._drop_count = 0
        self._human_state_counts = {state.name: 0 for state in HumanState}
        self._raw_contact_this_step = False
        self._accepted_contact_this_step = False
        self._acceptance_blocked_this_step = False
        self._contact_candidate_steps = 0
        self._last_transition_reason = "reset"
        self._last_features = InteractionFeatures()
        self._last_reward_context = RewardContext()
        self.readiness_config = HumanReadinessConfig.from_mapping(human_fsm_config)
        self._human_readiness = self.readiness_config.initial
        self._readiness_belief = self.readiness_config.initial
        self._allostatic_load_proxy = 0.0
        self._readiness_sum = 0.0
        self._readiness_min = self._human_readiness
        self._readiness_max = self._human_readiness
        self._readiness_blocked_count = 0
        self._readiness_crossed_threshold_this_step = False
        self._readiness_before_step = self._human_readiness
        self._readiness_hold_steps_remaining = 0
        self._reach_out_release_control_time = None
        self._animation_gated_by_readiness = False
        self._human_reach_progress = 0.0
        self._human_reach_progress_sum = 0.0
        self._reach_out_started_count = 0

        super().__init__(*args, **kwargs)
        if _HAS_HRGYM:
            self._compute_animation_time = self._compute_readiness_gated_animation_time

    def reset(self):
        self.allostatic_load.reset()
        self.human_fsm.reset()
        self._last_robot_speech = RobotSpeechToken.SILENCE
        self._previous_robot_speech = RobotSpeechToken.SILENCE
        self._last_human_speech = HumanSpeechEvent.SILENCE
        self._robot_speech_count = 0
        self._silence_count = 0
        self._repeated_speech_count = 0
        self._human_waiting_time = 0.0
        self._human_reach_effort_sum = 0.0
        self._handover_time = None
        self._episode_return = 0.0
        self._episode_step_count = 0
        self._load_sum = 0.0
        self._load_max = 0.0
        self._load_area = 0.0
        self._collision_count = 0
        self._drop_count = 0
        self._human_state_counts = {state.name: 0 for state in HumanState}
        self._raw_contact_this_step = False
        self._accepted_contact_this_step = False
        self._acceptance_blocked_this_step = False
        self._contact_candidate_steps = 0
        self._last_transition_reason = "reset"
        self._last_features = InteractionFeatures()
        self._last_reward_context = RewardContext()
        self._human_readiness = self.readiness_config.initial
        self._readiness_belief = self.readiness_config.initial
        self._allostatic_load_proxy = 0.0
        self._readiness_sum = 0.0
        self._readiness_min = self._human_readiness
        self._readiness_max = self._human_readiness
        self._readiness_blocked_count = 0
        self._readiness_crossed_threshold_this_step = False
        self._readiness_before_step = self._human_readiness
        self._readiness_hold_steps_remaining = 0
        self._reach_out_release_control_time = None
        self._animation_gated_by_readiness = False
        self._human_reach_progress = 0.0
        self._human_reach_progress_sum = 0.0
        self._reach_out_started_count = 0
        if self.readiness_config.enabled and self._human_readiness < self.readiness_config.threshold:
            self.human_fsm.force_state(HumanState.DISTRACTED)
        return super().reset()

    def set_robot_speech(self, token: RobotSpeechToken | int | str) -> None:
        if isinstance(token, str):
            token = RobotSpeechToken[token.strip().upper()]
        self._last_robot_speech = RobotSpeechToken(token)

    def set_robot_speech_from_scalar(self, value: float) -> RobotSpeechToken:
        token = robot_speech_from_scalar(value)
        self.set_robot_speech(token)
        return token

    def step(self, action):
        motor_action, speech_token = self._split_action(action)
        self.set_robot_speech(speech_token)
        if self._last_robot_speech != RobotSpeechToken.SILENCE:
            self._robot_speech_count += 1
            if self._last_robot_speech == self._previous_robot_speech:
                self._repeated_speech_count += 1
        else:
            self._silence_count += 1
        self._episode_step_count += 1

        self._raw_contact_this_step = False
        self._accepted_contact_this_step = False
        self._acceptance_blocked_this_step = False
        self._readiness_crossed_threshold_this_step = False
        self._advance_readiness_from_speech()
        self._sync_fsm_with_readiness_before_contact()

        obs, base_reward, done, info = super().step(motor_action)

        features = self._measure_interaction(obs=obs, info=info)
        load_snapshot = self.allostatic_load.update(features)
        success = bool(self._check_success(
            achieved_goal=self._get_achieved_goal_from_obs(obs),
            desired_goal=self._get_desired_goal_from_obs(obs),
        ))
        fsm_output = self.human_fsm.update(
            features=features,
            load_snapshot=load_snapshot,
            contact=self._accepted_contact_this_step,
            success=success,
        )
        self._last_human_speech = fsm_output.human_speech
        self._last_transition_reason = fsm_output.transition_reason
        self._sync_fsm_with_readiness_after_update()
        if self._readiness_crossed_threshold_this_step and self._last_human_speech == HumanSpeechEvent.SILENCE:
            self._last_human_speech = HumanSpeechEvent.READY
            self._last_transition_reason = "speech_cued_readiness"
        self._human_state_counts[self.human_fsm.state.name] += 1

        if features.forced_waiting > 0.0:
            self._human_waiting_time += features.forced_waiting
        self._human_reach_effort_sum += features.human_reach_effort
        if features.collision > 0.0:
            self._collision_count += 1
        load_total = float(load_snapshot["allostatic_load_total"])
        self._load_sum += load_total
        self._load_max = max(self._load_max, load_total)
        self._load_area += load_total * float(getattr(self, "control_timestep", 0.1))
        self._update_load_proxy_from_features()
        self._record_readiness_metrics()
        if success and self._handover_time is None:
            self._handover_time = float(getattr(self, "cur_time", getattr(self, "timestep", 0)))

        reward_context = RewardContext(
            allostatic_load_total=load_snapshot["allostatic_load_total"],
            forced_waiting=features.forced_waiting,
            proxemic_stress=load_snapshot["proxemic_stress"],
            human_reach_effort=features.human_reach_effort,
            robot_speech_count_step=1.0 if self._last_robot_speech != RobotSpeechToken.SILENCE else 0.0,
            uncomfortable_contact=features.uncomfortable_contact,
        )
        reward = compute_reward(base_reward, self.reward_variant, reward_context, self.reward_weights)
        self._episode_return += float(reward)
        self._last_features = features
        self._last_reward_context = reward_context

        info.update(self._get_allostatic_info(success=success, base_reward=base_reward, reward=reward))
        if done:
            info["episode_metrics"] = self._episode_metrics(success=success)

        self._previous_robot_speech = self._last_robot_speech
        return obs, reward, done, info

    def _split_action(self, action) -> tuple[np.ndarray, RobotSpeechToken]:
        arr = np.asarray(action, dtype=np.float32)
        if arr.ndim == 0:
            arr = arr.reshape(1)

        if self.speech_mode == "learned" and arr.size > 1:
            return arr[:-1], robot_speech_from_scalar(float(arr[-1]))
        return arr, self._last_robot_speech

    def _measure_interaction(self, obs: Mapping[str, Any], info: Mapping[str, Any]) -> InteractionFeatures:
        eef_pos = self._obs_vec(obs, self._eef_obs_key(), fallback=(0.0, 0.0, 0.0))
        object_pos = self._obs_vec(obs, "object_pos", fallback=eef_pos)
        target_pos = self._obs_vec(obs, "target_pos", fallback=object_pos)

        object_to_target = float(np.linalg.norm(object_pos - target_pos))
        eef_to_target = float(np.linalg.norm(eef_pos - target_pos))
        proximity_distance = min(object_to_target, eef_to_target)

        phase_name = getattr(getattr(self, "task_phase", None), "name", "")
        in_reach_out = phase_name == "REACH_OUT"
        forced_waiting = float(self.control_timestep if in_reach_out and not self._accepted_contact_this_step else 0.0)
        reach_effort = object_to_target
        collision = float(bool(info.get("collision", False)))
        uncomfortable_contact = float(self._acceptance_blocked_this_step or collision)
        uncertainty = float(
            in_reach_out
            and (
                (
                    self._last_robot_speech == RobotSpeechToken.SILENCE
                    and self.human_fsm.state in {HumanState.HESITANT, HumanState.DISTRACTED}
                )
                or (
                    self.readiness_config.enabled
                    and self._human_readiness < self.readiness_config.threshold
                )
            )
        )
        readiness_gain = max(0.0, self._human_readiness - self._readiness_before_step)
        smooth_progress = float(
            self._accepted_contact_this_step
            or self._last_robot_speech == RobotSpeechToken.REASSURE
            or readiness_gain > 0.0
        )

        return InteractionFeatures(
            robot_speech=self._last_robot_speech,
            previous_robot_speech=self._previous_robot_speech,
            forced_waiting=forced_waiting,
            proximity_distance=proximity_distance,
            human_reach_effort=reach_effort,
            uncertainty=uncertainty,
            uncomfortable_contact=uncomfortable_contact,
            collision=collision,
            smooth_progress=smooth_progress,
            step_seconds=float(getattr(self, "control_timestep", 0.1)),
        )

    def _get_object_palm_contact_pos(self, achieved_goal, desired_goal):
        contact_pos = super()._get_object_palm_contact_pos(achieved_goal=achieved_goal, desired_goal=desired_goal)
        self._raw_contact_this_step = contact_pos is not None
        if contact_pos is None:
            self._contact_candidate_steps = 0
            return None

        self._contact_candidate_steps += 1
        if self._readiness_allows_contact() or self.human_fsm.state == HumanState.GRASPING:
            self._accepted_contact_this_step = True
            self.human_fsm.force_state(HumanState.GRASPING)
            return contact_pos

        if self._readiness_adaptation_allows_contact():
            self._accepted_contact_this_step = True
            self.human_fsm.force_state(HumanState.GRASPING)
            return contact_pos

        self._acceptance_blocked_this_step = True
        self._readiness_blocked_count += 1
        return None

    def _setup_observables(self):
        observables = super()._setup_observables()
        if not _HAS_HRGYM:
            return observables

        @sensor(modality="speech")
        def previous_robot_speech_token(obs_cache):
            return np.array([float(self._previous_robot_speech.value)], dtype=np.float32)

        @sensor(modality="speech")
        def human_speech_event_token(obs_cache):
            return np.array([float(self._last_human_speech.value)], dtype=np.float32)

        @sensor(modality="allostatic_proxy")
        def human_readiness_belief(obs_cache):
            return np.array([float(self._readiness_belief)], dtype=np.float32)

        @sensor(modality="allostatic_proxy")
        def allostatic_load_proxy(obs_cache):
            return np.array([float(self._allostatic_load_proxy)], dtype=np.float32)

        sensors = [
            previous_robot_speech_token,
            human_speech_event_token,
            human_readiness_belief,
            allostatic_load_proxy,
        ]

        if self.privileged_observation:
            @sensor(modality="privileged")
            def human_state_token(obs_cache):
                return np.array([float(self.human_fsm.state.value)], dtype=np.float32)

            @sensor(modality="privileged")
            def allostatic_load_total(obs_cache):
                return np.array([float(self.allostatic_load.total)], dtype=np.float32)

            sensors.extend([human_state_token, allostatic_load_total])

        for sensor_fn in sensors:
            observables[sensor_fn.__name__] = Observable(
                name=sensor_fn.__name__,
                sensor=sensor_fn,
                sampling_rate=self.control_freq,
            )
        return observables

    def _get_allostatic_info(self, success: bool, base_reward: float, reward: float) -> dict[str, Any]:
        load = self.allostatic_load.snapshot()
        features = self._last_features
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

        info = {
            **load,
            **self.allostatic_load.last_delta,
            "reward_variant": self.reward_variant.value,
            "base_task_reward": float(base_reward),
            "allostatic_reward": float(reward),
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
            "human_reach_effort": features.human_reach_effort,
            "human_reach_effort_sum": self._human_reach_effort_sum,
            "allostatic_load_mean": self._mean_load(),
            "allostatic_load_max": self._load_max,
            "allostatic_load_final": load["allostatic_load_total"],
            "allostatic_load_area": self._load_area,
            "withdrawal_count": self.human_fsm.withdrawal_count,
            "overload_count": self.human_fsm.overload_count,
            "collision_count": self._collision_count,
            "drop_count": self._drop_count,
            "object_in_human_hand": bool(self._accepted_contact_this_step or success),
            "handover_time": self._handover_time,
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
            "animation_gated_by_readiness": self._animation_gated_by_readiness,
            "reach_out_started_count": self._reach_out_started_count,
            "success": success,
        }
        info.update(self._human_state_ratios())
        info.update(self._position_info())
        return info

    def _episode_metrics(self, success: bool) -> dict[str, Any]:
        load = self.allostatic_load.snapshot()
        metrics = {
            "success": float(success),
            "return": self._episode_return,
            "length": int(getattr(self, "timestep", 0)),
            "handover_time": self._handover_time,
            "robot_speech_count": self._robot_speech_count,
            "silence_count": self._silence_count,
            "silence_ratio": self._ratio(self._silence_count),
            "repeated_speech_count": self._repeated_speech_count,
            "human_waiting_time": self._human_waiting_time,
            "human_reach_effort": self._human_reach_effort_sum,
            "allostatic_load_mean": self._mean_load(),
            "allostatic_load_max": self._load_max,
            "allostatic_load_final": load["allostatic_load_total"],
            "allostatic_load_area": self._load_area,
            "withdrawal_count": self.human_fsm.withdrawal_count,
            "overload_count": self.human_fsm.overload_count,
            "collision_count": self._collision_count,
            "drop_count": self._drop_count,
            "object_in_human_hand": float(success),
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
            "animation_gated_by_readiness": self._animation_gated_by_readiness,
            "reach_out_started_count": self._reach_out_started_count,
            **load,
        }
        metrics.update(self._human_state_ratios())
        return metrics

    def _position_info(self) -> dict[str, Any]:
        try:
            eef_pos = np.asarray(self.sim.data.get_site_xpos(
                self.sim.model.site_id2name(self.robots[0].eef_site_id[self.robots[0].arms[0]])
            )).tolist()
            object_pos = np.asarray(self.sim.data.get_body_xpos(
                self.sim.model.body_id2name(self.manipulation_object_body_id)
            )).tolist()
            target_pos = np.asarray(self.target_pos).tolist()
            return {"robot_eef_pos": eef_pos, "object_pos": object_pos, "human_hand_pos": target_pos}
        except Exception:
            return {}

    def _eef_obs_key(self) -> str:
        try:
            return f"{self.robots[0].robot_model.naming_prefix}eef_pos"
        except Exception:
            return "robot0_eef_pos"

    @staticmethod
    def _obs_vec(obs: Mapping[str, Any], key: str, fallback) -> np.ndarray:
        value = obs.get(key, fallback)
        return np.asarray(value, dtype=np.float32).reshape(-1)[:3]

    def _mean_load(self) -> float:
        return self._load_sum / max(1, self._episode_step_count)

    def _mean_readiness(self) -> float:
        return self._readiness_sum / max(1, self._episode_step_count)

    def _mean_reach_progress(self) -> float:
        return self._human_reach_progress_sum / max(1, self._episode_step_count)

    def _ratio(self, count: int) -> float:
        return float(count) / max(1, self._episode_step_count)

    def _human_state_ratios(self) -> dict[str, float]:
        return {
            f"human_state_{state.name.lower()}_ratio": self._ratio(self._human_state_counts.get(state.name, 0))
            for state in HumanState
        }

    def _advance_readiness_from_speech(self) -> None:
        cfg = self.readiness_config
        if not cfg.enabled:
            self._human_readiness = 1.0
            self._readiness_belief = 1.0
            return

        self._readiness_before_step = self._human_readiness
        repeated = (
            self._last_robot_speech != RobotSpeechToken.SILENCE
            and self._last_robot_speech == self._previous_robot_speech
        )
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
        delta = effect - decay - load_drag
        self._human_readiness = self._clip_readiness(self._human_readiness + delta)
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
        self._readiness_crossed_threshold_this_step = (
            self._readiness_before_step < cfg.threshold <= self._human_readiness
        )

    def _compute_readiness_gated_animation_time(self, control_time: int) -> int:
        cfg = self.readiness_config
        if not cfg.enabled or not cfg.animation_gating:
            return RobotHumanHandoverCart._compute_animation_time(self, control_time)

        animation_time = int(control_time - self.animation_start_time)
        classic_animation_time = animation_time
        keyframes = self.human_animation_data[self.human_animation_id][1]["keyframes"]
        reach_start = int(keyframes[0])
        reach_end = int(keyframes[1])
        self._animation_gated_by_readiness = False

        if self.task_phase == RobotHumanHandoverPhase.APPROACH and animation_time > reach_start:
            if not self._readiness_allows_reach_out():
                self._reach_out_release_control_time = None
                self._animation_gated_by_readiness = True
                self._human_reach_progress = 0.0
                return reach_start

            if self._reach_out_release_control_time is None:
                self._reach_out_release_control_time = int(control_time)
                self._reach_out_started_count += 1
            animation_time = reach_start + max(0, int(control_time) - self._reach_out_release_control_time)
            classic_animation_time = animation_time
            self.task_phase = RobotHumanHandoverPhase.REACH_OUT

        if (
            self.task_phase == RobotHumanHandoverPhase.REACH_OUT
            and not self._readiness_allows_reach_out()
        ):
            self.task_phase = RobotHumanHandoverPhase.APPROACH
            self._reach_out_release_control_time = None
            self._animation_gated_by_readiness = True
            self._human_reach_progress = 0.0
            return reach_start

        if (
            animation_time > reach_start + (reach_end - reach_start) / 2
            and self.task_phase == RobotHumanHandoverPhase.REACH_OUT
        ):
            animation_time = int(
                layered_sin_modulations(
                    classic_animation_time=classic_animation_time,
                    modulation_start_time=(reach_start + reach_end) / 2,
                    amplitudes=self.animation_loop_amplitudes,
                    speeds=self.animation_loop_speeds,
                )
            )
            self._n_delayed_timesteps = classic_animation_time - animation_time

        if self.task_phase == RobotHumanHandoverPhase.RETREAT:
            animation_time -= self._n_delayed_timesteps

        if animation_time >= self.human_animation_length - 1:
            self.task_phase = RobotHumanHandoverPhase.COMPLETE
            animation_time = self.human_animation_length - 1

        self._human_reach_progress = self._compute_human_reach_progress(
            animation_time=animation_time,
            reach_start=reach_start,
            reach_end=reach_end,
        )
        return animation_time

    def _sync_fsm_with_readiness_before_contact(self) -> None:
        cfg = self.readiness_config
        if not cfg.enabled:
            return
        if self.human_fsm.state == HumanState.WITHDRAWING:
            return
        if self.task_phase == RobotHumanHandoverPhase.REACH_OUT and self._human_readiness < cfg.threshold:
            self.human_fsm.force_state(HumanState.DISTRACTED)
            return
        if self._human_readiness >= cfg.threshold and self.human_fsm.state in {
            HumanState.DISTRACTED,
            HumanState.HESITANT,
            HumanState.OVERLOADED,
        }:
            self.human_fsm.force_state(HumanState.READY)
        elif self._human_readiness < cfg.threshold and self.human_fsm.state == HumanState.READY:
            self.human_fsm.force_state(HumanState.DISTRACTED)

    def _sync_fsm_with_readiness_after_update(self) -> None:
        cfg = self.readiness_config
        if not cfg.enabled:
            return
        if self.human_fsm.state in {HumanState.WITHDRAWING, HumanState.GRASPING}:
            return
        if self.task_phase == RobotHumanHandoverPhase.REACH_OUT and self._human_readiness < cfg.threshold:
            self.human_fsm.force_state(HumanState.DISTRACTED)
            return
        if self._human_readiness >= cfg.threshold and self.allostatic_load.total < cfg.withdrawal_load_threshold:
            if self.human_fsm.state in {HumanState.DISTRACTED, HumanState.HESITANT, HumanState.OVERLOADED}:
                self.human_fsm.force_state(HumanState.READY)
        elif self._human_readiness < cfg.threshold and self.human_fsm.state == HumanState.READY:
            self.human_fsm.force_state(HumanState.DISTRACTED)

    def _readiness_allows_contact(self) -> bool:
        cfg = self.readiness_config
        if not cfg.enabled:
            return self.human_fsm.state in {HumanState.READY, HumanState.GRASPING}
        if self.human_fsm.state == HumanState.WITHDRAWING:
            return False
        if (
            cfg.animation_gating
            and self.human_fsm.state != HumanState.GRASPING
            and self.task_phase != RobotHumanHandoverPhase.REACH_OUT
        ):
            return False
        if cfg.animation_gating and self._human_reach_progress < cfg.min_reach_progress_for_contact:
            return False
        return self._human_readiness >= cfg.threshold

    def _readiness_adaptation_allows_contact(self) -> bool:
        cfg = self.readiness_config
        if not cfg.enabled:
            return False
        if self.human_fsm.state == HumanState.WITHDRAWING:
            return False
        if self._contact_candidate_steps < self.acceptance_delay_steps:
            return False
        if cfg.animation_gating and self._human_reach_progress < cfg.min_reach_progress_for_contact:
            return False
        return self._human_readiness >= cfg.adaptive_threshold

    def _readiness_allows_reach_out(self) -> bool:
        cfg = self.readiness_config
        if not cfg.enabled:
            return True
        if self.human_fsm.state == HumanState.WITHDRAWING:
            return False
        return self._human_readiness >= cfg.threshold

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
    def _compute_human_reach_progress(animation_time: int, reach_start: int, reach_end: int) -> float:
        span = max(1, reach_end - reach_start)
        return max(0.0, min(1.0, float(animation_time - reach_start) / float(span)))

    @staticmethod
    def _clip_readiness(value: float) -> float:
        return max(0.0, min(1.0, float(value)))


@dataclass
class HumanReadinessConfig:
    enabled: bool = True
    initial: float = 0.20
    threshold: float = 0.55
    adaptive_threshold: float = 0.42
    decay: float = 0.004
    belief_decay: float = 0.003
    repeated_effect_scale: float = 0.65
    load_resilience_threshold: float = 7.0
    withdrawal_load_threshold: float = 9.0
    load_drag: float = 0.01
    max_load_drag: float = 0.04
    animation_gating: bool = True
    min_reach_progress_for_contact: float = 0.65
    readiness_hold_steps: int = 160
    readiness_hold_floor: float = 0.72
    readiness_hold_tokens: tuple[RobotSpeechToken, ...] = field(
        default_factory=lambda: (
            RobotSpeechToken.ANNOUNCE_HANDOVER,
            RobotSpeechToken.SAY_RELEASING,
        )
    )
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

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "HumanReadinessConfig":
        if values is None:
            return cls()
        source = dict(values)
        if isinstance(source.get("readiness"), dict):
            source = dict(source["readiness"])
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        kwargs = {
            key: value
            for key, value in source.items()
            if key in allowed and key not in {"speech_effects", "readiness_hold_tokens"}
        }
        config = cls(**kwargs)
        hold_tokens = source.get("readiness_hold_tokens")
        if isinstance(hold_tokens, str):
            hold_tokens = [hold_tokens]
        if isinstance(hold_tokens, (list, tuple, set)):
            config.readiness_hold_tokens = tuple(
                RobotSpeechToken[item.strip().upper()] if isinstance(item, str) else RobotSpeechToken(item)
                for item in hold_tokens
            )
        speech_effects = source.get("speech_effects")
        if isinstance(speech_effects, Mapping):
            effects = dict(config.speech_effects)
            for key, value in speech_effects.items():
                token = RobotSpeechToken[key.strip().upper()] if isinstance(key, str) else RobotSpeechToken(key)
                effects[token] = float(value)
            config.speech_effects = effects
        return config
