"""human-robot-gym training wrapper stacks."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from allostatic_handover.envs.speech_events import RobotSpeechToken, robot_speech_from_scalar
from allostatic_handover.wrappers.robosuite_gymnasium import (
    ORIGINAL_HANDOVER_OBS_KEYS,
    adapt_robosuite_for_sb3,
)


def make_sb3_env_from_hrgym(
    raw_env,
    handover_env: str = "allostatic",
    wrapper_stack: str = "raw",
    expert_imitation_alpha: float = 0.25,
    expert_imitation_beta: float = 0.7,
):
    """Wrap a human-robot-gym / robosuite env for SB3 training."""
    wrapper_stack = wrapper_stack.strip().lower()
    handover_env = handover_env.strip().lower()
    obs_keys = _obs_keys_for_handover_env(handover_env, raw_env=raw_env)
    append_speech_action = handover_env == "allostatic"

    if wrapper_stack == "raw":
        return adapt_robosuite_for_sb3(
            raw_env,
            obs_keys=None if append_speech_action else obs_keys,
            append_speech_action=append_speech_action,
        )

    if wrapper_stack == "safe_ik":
        if append_speech_action:
            _set_external_speech_mode(raw_env)
            env = _apply_safe_ik_stack(raw_env=raw_env, obs_keys=obs_keys)
            env = SpeechActionWrapper(env=env, speech_env=raw_env)
            return adapt_robosuite_for_sb3(
                env,
                obs_keys=None,
                append_speech_action=False,
                force_adapter=True,
            )
        env = _apply_safe_ik_stack(raw_env=raw_env, obs_keys=obs_keys)
        return adapt_robosuite_for_sb3(
            env,
            obs_keys=None,
            append_speech_action=False,
        )

    if wrapper_stack == "safe_ik_air":
        if append_speech_action:
            _set_external_speech_mode(raw_env)
            env = _apply_safe_ik_stack(
                raw_env=raw_env,
                obs_keys=obs_keys,
                expert_obs_keys=EXPERT_HANDOVER_OBS_KEYS,
            )
            env = _apply_action_based_expert_reward(
                env=env,
                expert_imitation_alpha=expert_imitation_alpha,
                expert_imitation_beta=expert_imitation_beta,
            )
            env = SpeechActionWrapper(env=env, speech_env=raw_env)
            return adapt_robosuite_for_sb3(
                env,
                obs_keys=None,
                append_speech_action=False,
                force_adapter=True,
            )
        env = _apply_safe_ik_stack(
            raw_env=raw_env,
            obs_keys=obs_keys,
            expert_obs_keys=EXPERT_HANDOVER_OBS_KEYS,
        )
        env = _apply_action_based_expert_reward(
            env=env,
            expert_imitation_alpha=expert_imitation_alpha,
            expert_imitation_beta=expert_imitation_beta,
        )
        return adapt_robosuite_for_sb3(
            env,
            obs_keys=None,
            append_speech_action=False,
            force_adapter=True,
        )

    raise ValueError("wrapper_stack must be 'raw', 'safe_ik', or 'safe_ik_air'")


EXPERT_HANDOVER_OBS_KEYS = (
    "object_gripped",
    "vec_eef_to_object",
    "vec_eef_to_target",
    "robot0_gripper_qpos",
)

ALLOSTATIC_HANDOVER_OBS_KEYS = ORIGINAL_HANDOVER_OBS_KEYS + (
    "previous_robot_speech_token",
    "human_speech_event_token",
    "human_readiness_belief",
    "allostatic_load_proxy",
)


def _obs_keys_for_handover_env(handover_env: str, raw_env=None) -> Iterable[str] | None:
    handover_env = handover_env.strip().lower()
    if handover_env == "original":
        return ORIGINAL_HANDOVER_OBS_KEYS
    if handover_env == "allostatic":
        keys = list(ALLOSTATIC_HANDOVER_OBS_KEYS)
        if getattr(raw_env, "privileged_observation", False):
            keys.extend(["human_state_token", "allostatic_load_total"])
        return tuple(keys)
    return None


def _set_external_speech_mode(raw_env) -> None:
    if hasattr(raw_env, "speech_mode"):
        raw_env.speech_mode = "external"


class SpeechActionWrapper:
    """Split a 5D policy action into 4D safe-IK motor action plus speech.

    The wrapped safe-IK stack should see only ``(dx, dy, dz, gripper)``. The
    final scalar is decoded to a speech token and injected into the underlying
    allostatic robosuite environment before the motor step reaches it.
    """

    def __init__(self, env, speech_env=None):
        self.env = env
        self.speech_env = speech_env if speech_env is not None else getattr(env, "unwrapped", env)
        low, high = _low_high_from_env(env)
        self._motor_action_dim = int(low.reshape(-1).shape[0])
        self._low = np.concatenate([low.reshape(-1), np.array([-1.0], dtype=np.float32)]).astype(np.float32)
        self._high = np.concatenate([high.reshape(-1), np.array([1.0], dtype=np.float32)]).astype(np.float32)
        self.action_space = _box_space(self._low, self._high)
        self.observation_space = getattr(env, "observation_space", None)
        self.reward_range = getattr(env, "reward_range", None)
        self.metadata = getattr(env, "metadata", None)
        self.last_robot_speech = RobotSpeechToken.SILENCE

    @property
    def unwrapped(self):
        if hasattr(self.env, "unwrapped"):
            return self.env.unwrapped
        return self.env

    @property
    def action_spec(self):
        return self._low.copy(), self._high.copy()

    def reset(self, *args, **kwargs):
        self.last_robot_speech = RobotSpeechToken.SILENCE
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size != self._motor_action_dim + 1:
            raise ValueError(
                f"Expected {self._motor_action_dim + 1}D action "
                f"(motor + speech), got shape {arr.shape}."
            )
        motor_action = arr[: self._motor_action_dim]
        token = robot_speech_from_scalar(float(arr[-1]))
        self.last_robot_speech = token
        if hasattr(self.speech_env, "set_robot_speech"):
            self.speech_env.set_robot_speech(token)
        return self.env.step(motor_action)

    def render(self, **kwargs):
        if hasattr(self.env, "render"):
            return self.env.render(**kwargs)
        return None

    def close(self):
        if hasattr(self.env, "close"):
            return self.env.close()
        return None

    def __getattr__(self, attr: str):
        return getattr(self.env, attr)


def _low_high_from_env(env) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(env, "action_spec"):
        low, high = env.action_spec
    elif hasattr(env, "action_space"):
        low = env.action_space.low
        high = env.action_space.high
    else:
        raise AttributeError("Wrapped environment must expose action_spec or action_space")
    return np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32)


def _box_space(low: np.ndarray, high: np.ndarray):
    try:
        from gymnasium import spaces

        return spaces.Box(low=low, high=high, dtype=np.float32)
    except ImportError:
        class _Box:
            def __init__(self, low, high):
                self.low = low
                self.high = high
                self.shape = low.shape

            def sample(self):
                return np.random.uniform(self.low, self.high).astype(np.float32)

        return _Box(low=low, high=high)


def _apply_safe_ik_stack(
    raw_env,
    obs_keys: Iterable[str] | None,
    expert_obs_keys: Iterable[str] | None = None,
):
    from robosuite.wrappers import GymWrapper
    from human_robot_gym.utils.mjcf_utils import file_path_completion
    from human_robot_gym.wrappers.collision_prevention_wrapper import CollisionPreventionWrapper
    from human_robot_gym.wrappers.ik_position_delta_wrapper import IKPositionDeltaWrapper

    gym_env = (
        _ExpertObsCompatWrapper(raw_env, agent_keys=obs_keys, expert_keys=expert_obs_keys)
        if expert_obs_keys is not None
        else GymWrapper(raw_env, keys=obs_keys)
    )
    env = CollisionPreventionWrapper(
        env=gym_env,
        collision_check_fn=raw_env.check_collision_action,
        replace_type=0,
        n_resamples=20,
    )
    env = IKPositionDeltaWrapper(
        env=env,
        urdf_file=file_path_completion("models/assets/robots/schunk/robot_pybullet.urdf"),
        action_limits=np.array([[-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]], dtype=np.float32),
        x_output_max=1.0,
        x_position_limits=None,
        residual_threshold=1e-3,
        max_iter=50,
    )
    _set_action_space_from_spec(env)
    return env


def _apply_action_based_expert_reward(env, expert_imitation_alpha: float, expert_imitation_beta: float):
    return _ActionBasedExpertRewardCompatWrapper(
        env=env,
        expert=_ScalarPickPlaceHumanCartExpert(
            observation_space=env.observation_space,
            action_space=env.action_space,
            signal_to_noise_ratio=0.98,
            hover_dist=0.2,
            tan_theta=0.5,
            horizontal_epsilon=0.035,
            vertical_epsilon=0.015,
            goal_dist=0.08,
            gripper_fully_opened_threshold=0.02,
            release_when_delivered=True,
            delta_time=0.01,
        ),
        alpha=expert_imitation_alpha,
        beta=expert_imitation_beta,
        iota_m=0.1,
        iota_g=0.5,
        m_sim_fn="gaussian",
        g_sim_fn="gaussian",
    )


class _ExpertObsCompatWrapper:
    """Flatten agent observations and keep expert observations in ``info``.

    human-robot-gym's ExpertObsWrapper expects a five-value step API. The
    currently installed RobotHumanHandoverCart returns the older four-value
    robosuite API, so this local wrapper keeps the same semantics while
    accepting both forms.
    """

    PREVIOUS_EXPERT_OBSERVATION_KEY = "previous_expert_observation"
    CURRENT_EXPERT_OBSERVATION_KEY = "current_expert_observation"

    def __init__(
        self,
        env,
        agent_keys: Iterable[str] | None,
        expert_keys: Iterable[str] | None,
    ):
        from gymnasium import spaces

        self.env = env
        self.agent_keys = tuple(agent_keys or ())
        self.expert_keys = tuple(expert_keys or ())
        self.reward_range = (0, getattr(env, "reward_scale", 1.0))
        self.name = f"{type(env).__name__}_ExpertObsCompat"
        self.metadata = None

        obs_dict = env.reset()
        flat_ob = self._flatten_obs(obs_dict)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=flat_ob.shape,
            dtype=np.float32,
        )
        low, high = env.action_spec
        self.action_space = spaces.Box(low=np.asarray(low, dtype=np.float32), high=np.asarray(high, dtype=np.float32))
        self._current_expert_observation = self._filter_expert_obs(obs_dict)
        self._previous_expert_observation = None

    @property
    def unwrapped(self):
        if hasattr(self.env, "unwrapped"):
            return self.env.unwrapped
        return self.env

    @property
    def action_spec(self):
        return self.env.action_spec

    def reset(self):
        obs_dict = self.env.reset()
        self._previous_expert_observation = None
        self._current_expert_observation = self._filter_expert_obs(obs_dict)
        return self._flatten_obs(obs_dict)

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs_dict, reward, terminated, truncated, info = result
        else:
            obs_dict, reward, done, info = result
            truncated = bool(info.get("timeout", False))
            terminated = bool(done and not truncated)

        self._previous_expert_observation = self._current_expert_observation
        self._current_expert_observation = self._filter_expert_obs(obs_dict)
        info[self.PREVIOUS_EXPERT_OBSERVATION_KEY] = self._previous_expert_observation
        info[self.CURRENT_EXPERT_OBSERVATION_KEY] = self._current_expert_observation
        return self._flatten_obs(obs_dict), reward, bool(terminated), bool(truncated), info

    def close(self):
        if hasattr(self.env, "close"):
            return self.env.close()
        return None

    def render(self, **kwargs):
        if hasattr(self.env, "render"):
            return self.env.render(**kwargs)
        return None

    def __getattr__(self, attr: str):
        return getattr(self.env, attr)

    def _filter_expert_obs(self, obs_dict: dict[str, Any]) -> dict[str, Any]:
        return {key: obs_dict[key] for key in self.expert_keys if key in obs_dict}

    def _flatten_obs(self, obs_dict: dict[str, Any]) -> np.ndarray:
        parts = []
        missing = []
        for key in self.agent_keys:
            if key not in obs_dict:
                missing.append(key)
                continue
            parts.append(np.asarray(obs_dict[key], dtype=np.float32).reshape(-1))
        if missing:
            raise KeyError(f"Observation keys missing from robosuite obs: {missing}")
        return np.concatenate(parts).astype(np.float32, copy=False)


class _ScalarPickPlaceHumanCartExpert:
    """PickPlaceHumanCart expert with scalar gripper output for NumPy 2 / Python 3.13."""

    def __init__(self, *args, **kwargs):
        from human_robot_gym.demonstrations.experts.pick_place_human_cart_expert import PickPlaceHumanCartExpert

        self._expert = PickPlaceHumanCartExpert(*args, **kwargs)
        self.action_space = self._expert.action_space
        self.observation_space = self._expert.observation_space

    def __call__(self, obs_dict: dict[str, Any]) -> np.ndarray:
        obs = self._expert.expert_observation_from_dict(obs_dict=obs_dict)
        action = np.zeros(4, dtype=np.float32)
        motion = self._expert._select_motion(obs).clip(
            -self._expert._motion_action_limit,
            self._expert._motion_action_limit,
        )
        action[:3] = (
            motion * self._expert._signal_to_noise_ratio
            + self._expert._motion_noise.step(dt=self._expert._delta_time)
            * (1 - self._expert._signal_to_noise_ratio)
        ).clip(-self._expert._motion_action_limit, self._expert._motion_action_limit)
        action[3] = float(np.asarray(self._expert._select_gripper_action(obs)).reshape(-1)[0])
        return action


class _ActionBasedExpertRewardCompatWrapper:
    """Cartesian action imitation reward wrapper compatible with robosuite wrappers."""

    def __init__(
        self,
        env,
        expert,
        alpha: float,
        beta: float,
        iota_m: float,
        iota_g: float,
        m_sim_fn: str,
        g_sim_fn: str,
    ):
        from gymnasium import spaces

        self.env = env
        self._expert = expert
        self._alpha = float(alpha)
        self._beta = float(beta)
        self._iota_m = float(iota_m)
        self._iota_g = float(iota_g)
        self._m_sim_fn = str(m_sim_fn)
        self._g_sim_fn = str(g_sim_fn)
        self.observation_space = env.observation_space
        low, high = env.action_spec
        self.action_space = spaces.Box(low=np.asarray(low, dtype=np.float32), high=np.asarray(high, dtype=np.float32))
        self._imitation_rewards: list[float] = []
        self._environment_rewards: list[float] = []

    @property
    def unwrapped(self):
        if hasattr(self.env, "unwrapped"):
            return self.env.unwrapped
        return self.env

    @property
    def action_spec(self):
        return self.env.action_spec

    def reset(self):
        self._imitation_rewards = []
        self._environment_rewards = []
        return self.env.reset()

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        expert_obs = info[_ExpertObsCompatWrapper.PREVIOUS_EXPERT_OBSERVATION_KEY]
        expert_action = self._expert(expert_obs)
        imitation_reward = self._get_imitation_reward(np.asarray(action), expert_action)
        reward = self._combine_reward(env_reward, imitation_reward)

        self._imitation_rewards.append(float(imitation_reward))
        self._environment_rewards.append(float(env_reward))
        info["imitation_reward"] = float(imitation_reward)
        info["environment_reward"] = float(env_reward)
        if terminated or truncated:
            self._add_reward_to_info(info)
        return obs, float(reward), terminated, truncated, info

    def close(self):
        if hasattr(self.env, "close"):
            return self.env.close()
        return None

    def render(self, **kwargs):
        if hasattr(self.env, "render"):
            return self.env.render(**kwargs)
        return None

    def __getattr__(self, attr: str):
        return getattr(self.env, attr)

    def _get_imitation_reward(self, agent_action: np.ndarray, expert_action: np.ndarray) -> float:
        from human_robot_gym.utils.expert_imitation_reward_utils import similarity_fn

        motion_imitation_reward = similarity_fn(
            name=self._m_sim_fn,
            delta=np.linalg.norm(agent_action[:3] - expert_action[:3]),
            iota=self._iota_m,
        )
        gripper_imitation_reward = similarity_fn(
            name=self._g_sim_fn,
            delta=np.abs(agent_action[3] - expert_action[3]),
            iota=self._iota_g,
        )
        return float(motion_imitation_reward * self._beta + gripper_imitation_reward * (1 - self._beta))

    def _combine_reward(self, env_reward: float, imitation_reward: float) -> float:
        return float(imitation_reward * self._alpha + env_reward * (1 - self._alpha))

    def _add_reward_to_info(self, info: dict[str, Any]) -> None:
        ep_im_rew = float(np.sum(self._imitation_rewards))
        ep_env_rew = float(np.sum(self._environment_rewards))
        info["ep_im_rew_mean"] = ep_im_rew
        info["ep_env_rew_mean"] = ep_env_rew
        info["ep_full_rew_mean"] = self._combine_reward(ep_env_rew, ep_im_rew)
        info["im_rew_mean"] = float(np.mean(self._imitation_rewards)) if self._imitation_rewards else np.nan
        info["env_rew_mean"] = float(np.mean(self._environment_rewards)) if self._environment_rewards else np.nan
        info["full_rew_mean"] = self._combine_reward(info["env_rew_mean"], info["im_rew_mean"])


def _set_action_space_from_spec(env) -> None:
    from gymnasium import spaces

    low, high = env.action_spec
    env.action_space = spaces.Box(
        low=np.asarray(low, dtype=np.float32),
        high=np.asarray(high, dtype=np.float32),
        dtype=np.float32,
    )
