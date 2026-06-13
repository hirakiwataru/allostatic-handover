"""Gymnasium adapter for robosuite-style environments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - imported only for training extras
    gym = None
    spaces = None


DEFAULT_OBS_KEYS = (
    "robot0_proprio-state",
    "object-state",
    "goal-state",
    "speech-state",
)

ORIGINAL_HANDOVER_OBS_KEYS = (
    "object_gripped",
    "vec_eef_to_object",
    "vec_eef_to_target",
    "gripper_aperture",
    "dist_eef_to_human_head",
    "dist_eef_to_human_lh",
    "dist_eef_to_human_rh",
)


class RobosuiteGymnasiumAdapter(gym.Env if gym is not None else object):
    """Expose a robosuite env as a Gymnasium Box-observation environment.

    human-robot-gym / robosuite environments expose ``action_spec`` and dict
    observations, while SB3 expects Gymnasium spaces. This adapter keeps the
    underlying MuJoCo environment intact and only flattens selected observation
    keys plus appends the learned speech scalar to the motor action.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env,
        obs_keys: Iterable[str] | None = None,
        append_speech_action: bool = True,
        force_adapter: bool = False,
    ):
        if gym is None or spaces is None:
            raise ImportError("gymnasium is required for RobosuiteGymnasiumAdapter")

        self.env = env
        self.obs_keys = tuple(obs_keys or DEFAULT_OBS_KEYS)
        self.append_speech_action = bool(append_speech_action)
        self.force_adapter = bool(force_adapter)
        self._last_info: dict[str, Any] = {}

        if hasattr(env, "action_spec"):
            low, high = env.action_spec
        elif hasattr(env, "action_space"):
            low = env.action_space.low
            high = env.action_space.high
        else:
            raise AttributeError("Wrapped environment must expose action_spec or action_space")
        low = np.asarray(low, dtype=np.float32).reshape(-1)
        high = np.asarray(high, dtype=np.float32).reshape(-1)
        if self.append_speech_action:
            low = np.concatenate([low, np.array([-1.0], dtype=np.float32)])
            high = np.concatenate([high, np.array([1.0], dtype=np.float32)])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        sample_obs = self._reset_raw()
        flat_obs = self._flatten_obs(sample_obs)
        obs_bound = np.full(flat_obs.shape, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_bound, high=obs_bound, dtype=np.float32)

    @property
    def unwrapped(self):
        return self.env

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del seed, options
        self._last_info = {}
        obs = self._reset_raw()
        return self._flatten_obs(obs), {}

    def step(self, action):
        result = self.env.step(np.asarray(action, dtype=np.float32).reshape(-1))
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            truncated = bool(truncated or info.get("timeout", False))
        else:
            obs, reward, done, info = result
            terminated = bool(done)
            truncated = bool(info.get("timeout", False))
        self._last_info = dict(info)
        return self._flatten_obs(obs), float(reward), bool(terminated), bool(truncated), info

    def render(self):
        if hasattr(self.env, "render"):
            return self.env.render()
        return None

    def close(self):
        if hasattr(self.env, "close"):
            return self.env.close()
        return None

    def _reset_raw(self):
        result = self.env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
            self._last_info = dict(info)
            return obs
        return result

    def _flatten_obs(self, obs) -> np.ndarray:
        if isinstance(obs, Mapping):
            parts = []
            missing = []
            for key in self.obs_keys:
                if key not in obs:
                    missing.append(key)
                    continue
                parts.append(np.asarray(obs[key], dtype=np.float32).reshape(-1))
            if missing:
                raise KeyError(f"Observation keys missing from robosuite obs: {missing}")
            return np.concatenate(parts).astype(np.float32, copy=False)
        return np.asarray(obs, dtype=np.float32).reshape(-1)


def adapt_robosuite_for_sb3(
    env,
    obs_keys: Iterable[str] | None = None,
    append_speech_action: bool | None = None,
    force_adapter: bool = False,
):
    """Return a Gymnasium adapter only when the env is not already Gymnasium."""
    if not force_adapter and gym is not None and isinstance(env, gym.Env):
        return env
    if append_speech_action is None:
        append_speech_action = getattr(env, "speech_mode", None) == "learned"
    return RobosuiteGymnasiumAdapter(
        env,
        obs_keys=obs_keys,
        append_speech_action=append_speech_action,
        force_adapter=force_adapter,
    )
