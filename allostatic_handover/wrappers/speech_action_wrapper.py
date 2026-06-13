"""Gymnasium action wrapper that appends a speech scalar to Box actions."""

from __future__ import annotations

import numpy as np


try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None


class SpeechActionWrapper(gym.ActionWrapper if gym is not None else object):
    """Append one scalar action and send it to the raw allostatic env as speech."""

    def __init__(self, env):
        if gym is None:
            raise ImportError("gymnasium is required for SpeechActionWrapper")
        super().__init__(env)
        if not isinstance(env.action_space, spaces.Box):
            raise TypeError("SpeechActionWrapper only supports Box action spaces.")
        low = np.concatenate([env.action_space.low.reshape(-1), np.array([-1.0], dtype=np.float32)])
        high = np.concatenate([env.action_space.high.reshape(-1), np.array([1.0], dtype=np.float32)])
        self._base_shape = env.action_space.shape
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def action(self, action):
        flat = np.asarray(action, dtype=np.float32).reshape(-1)
        speech_scalar = float(flat[-1])
        raw = getattr(self.env, "env", self.env)
        if hasattr(raw, "set_robot_speech_from_scalar"):
            raw.set_robot_speech_from_scalar(speech_scalar)
        return flat[:-1].reshape(self._base_shape)
