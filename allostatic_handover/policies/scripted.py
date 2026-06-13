"""Scripted policies used before RL training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from allostatic_handover.envs.speech_events import RobotSpeechToken, robot_speech_to_scalar


def _action_shape(env) -> tuple[int, ...]:
    space = getattr(env, "action_space", None)
    if space is not None and getattr(space, "shape", None) is not None:
        return tuple(space.shape)
    try:
        low, _high = env.action_spec
        base_shape = tuple(np.asarray(low).shape)
        if getattr(env, "speech_mode", "") == "learned" and len(base_shape) == 1:
            return (base_shape[0] + 1,)
        return base_shape
    except Exception:
        return (5,)


def _zero_action(env) -> np.ndarray:
    return np.zeros(_action_shape(env), dtype=np.float32)


def _set_speech(action: np.ndarray, token: RobotSpeechToken) -> np.ndarray:
    if action.size == 0:
        return action
    action = action.copy()
    action.reshape(-1)[-1] = robot_speech_to_scalar(token)
    return action


def _mock_motion(obs: Any, action: np.ndarray) -> np.ndarray:
    """Simple controller for MockAllostaticHandoverEnv."""
    if isinstance(obs, dict):
        return action
    flat = np.asarray(obs, dtype=np.float32).reshape(-1)
    if flat.size < 7 or action.size < 5:
        return action
    robot = flat[0:2]
    obj = flat[2:4]
    human = flat[4:6]
    gripped = bool(flat[6] > 0.5)
    target = human if gripped else obj
    delta = np.clip((target - robot) * 8.0, -1.0, 1.0)
    action = action.copy()
    action[0:2] = delta
    action[3] = 1.0 if not gripped else 0.8
    if gripped and np.linalg.norm(obj - human) < 0.1:
        action[3] = -1.0
    return action


@dataclass
class ScriptedPolicyState:
    step: int = 0
    release_cued: bool = False


class BaseScriptedPolicy:
    name = "base"

    def __init__(self):
        self.state = ScriptedPolicyState()

    def reset(self) -> None:
        self.state = ScriptedPolicyState()

    def speech(self, obs: Any, info: dict[str, Any] | None) -> RobotSpeechToken:
        return RobotSpeechToken.SILENCE

    def act(self, obs: Any, env, info: dict[str, Any] | None = None) -> np.ndarray:
        action = _mock_motion(obs, _zero_action(env))
        action = _set_speech(action, self.speech(obs, info))
        self.state.step += 1
        return action


class MinimalSpeechPolicy(BaseScriptedPolicy):
    name = "minimal_speech"

    def speech(self, obs: Any, info: dict[str, Any] | None) -> RobotSpeechToken:
        if self.state.step == 0:
            return RobotSpeechToken.ANNOUNCE_HANDOVER
        if info and not self.state.release_cued and info.get("human_reach_effort", 1.0) < 0.12:
            self.state.release_cued = True
            return RobotSpeechToken.SAY_RELEASING
        return RobotSpeechToken.SILENCE


class ExcessiveSpeechPolicy(BaseScriptedPolicy):
    name = "excessive_speech"

    def speech(self, obs: Any, info: dict[str, Any] | None) -> RobotSpeechToken:
        sequence = [
            RobotSpeechToken.ASK_READY,
            RobotSpeechToken.ASK_READY,
            RobotSpeechToken.REASSURE,
            RobotSpeechToken.ASK_CONFIRMATION,
        ]
        return sequence[self.state.step % len(sequence)]


class HumanWaitingPolicy(BaseScriptedPolicy):
    name = "human_waiting"

    def speech(self, obs: Any, info: dict[str, Any] | None) -> RobotSpeechToken:
        return RobotSpeechToken.ASK_READY if self.state.step % 2 == 0 else RobotSpeechToken.SAY_WAITING

    def act(self, obs: Any, env, info: dict[str, Any] | None = None) -> np.ndarray:
        action = _zero_action(env)
        if action.size >= 5:
            action[3] = 1.0
        action = _set_speech(action, self.speech(obs, info))
        self.state.step += 1
        return action


class AllostaticAwareScriptedPolicy(BaseScriptedPolicy):
    name = "allostatic_aware"

    def speech(self, obs: Any, info: dict[str, Any] | None) -> RobotSpeechToken:
        if self.state.step == 0:
            return RobotSpeechToken.ANNOUNCE_HANDOVER
        if info:
            human_state = str(info.get("human_state", "READY"))
            load = float(info.get("allostatic_load_total", 0.0))
            if human_state in {"HESITANT", "OVERLOADED"} or load > 2.0:
                return RobotSpeechToken.REASSURE
            if not self.state.release_cued and info.get("human_reach_effort", 1.0) < 0.10:
                self.state.release_cued = True
                return RobotSpeechToken.SAY_RELEASING
        return RobotSpeechToken.SILENCE


class RandomPolicy(BaseScriptedPolicy):
    name = "random"

    def act(self, obs: Any, env, info: dict[str, Any] | None = None) -> np.ndarray:
        space = getattr(env, "action_space", None)
        if space is not None and hasattr(space, "sample"):
            action = np.asarray(space.sample(), dtype=np.float32)
        else:
            action = np.random.uniform(-1.0, 1.0, size=_action_shape(env)).astype(np.float32)
        self.state.step += 1
        return action


def make_scripted_policy(name: str) -> BaseScriptedPolicy:
    key = name.strip().lower().replace("-", "_")
    policies = {
        "minimal": MinimalSpeechPolicy,
        "minimal_speech": MinimalSpeechPolicy,
        "excessive": ExcessiveSpeechPolicy,
        "excessive_speech": ExcessiveSpeechPolicy,
        "waiting": HumanWaitingPolicy,
        "human_waiting": HumanWaitingPolicy,
        "allostatic": AllostaticAwareScriptedPolicy,
        "allostatic_aware": AllostaticAwareScriptedPolicy,
        "random": RandomPolicy,
    }
    if key not in policies:
        raise ValueError(f"Unknown scripted policy '{name}'. Choose one of {sorted(policies)}")
    return policies[key]()
