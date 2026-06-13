"""Fallback stub for human-robot-gym when sara-shield bindings are unavailable.

This is only intended for `shield_type="OFF"` smoke tests. It lets the
human-robot-gym environments import and run without the compiled SaRA safety
shield. Do not use this as a replacement for safety evaluation.
"""

from __future__ import annotations

from enum import Enum

import numpy as np


class ShieldType(Enum):
    OFF = 0
    SSM = 1
    PFL = 2


class ContactType(Enum):
    WEDGE = 0


class AABB:
    def __init__(self, lower, upper):
        self.lower = list(lower)
        self.upper = list(upper)


class _DesiredMotion:
    def __init__(self, qpos):
        self._qpos = np.asarray(qpos, dtype=float)

    def getAngle(self):
        return self._qpos

    def getVelocity(self):
        return np.zeros_like(self._qpos)

    def getAcceleration(self):
        return np.zeros_like(self._qpos)


class SafetyShield:
    def __init__(self, init_qpos=None, **kwargs):
        self._qpos = np.asarray(init_qpos if init_qpos is not None else [], dtype=float)
        self._safe = True

    def reset(self, init_qpos=None, **kwargs):
        if init_qpos is not None:
            self._qpos = np.asarray(init_qpos, dtype=float)
        self._safe = True

    def step(self, current_time):
        return _DesiredMotion(self._qpos)

    def newLongTermTrajectory(self, goal_qpos, command_vel):
        self._qpos = np.asarray(goal_qpos, dtype=float)

    def humanMeasurement(self, human_measurement, time):
        return None

    def getSafety(self):
        return self._safe

    def getRobotReachCapsules(self):
        return []

    def getHumanReachCapsules(self, index):
        return []
