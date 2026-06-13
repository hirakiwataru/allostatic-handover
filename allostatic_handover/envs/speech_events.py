"""Discrete speech tokens used by the MVP.

The experiment intentionally models speech as symbolic tokens. TTS, ASR, and
waveforms are outside the current scope.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Iterable


class RobotSpeechToken(IntEnum):
    SILENCE = 0
    ANNOUNCE_HANDOVER = 1
    ASK_READY = 2
    REASSURE = 3
    SAY_WAITING = 4
    SAY_RELEASING = 5
    ASK_CONFIRMATION = 6


class HumanSpeechEvent(IntEnum):
    SILENCE = 0
    READY = 1
    WAIT = 2
    CLOSER = 3
    SLOWER = 4
    GOT_IT = 5
    TOO_CLOSE = 6
    CONFUSED = 7


ROBOT_SPEECH_TEXT = {
    RobotSpeechToken.SILENCE: "",
    RobotSpeechToken.ANNOUNCE_HANDOVER: "今から渡します",
    RobotSpeechToken.ASK_READY: "準備できましたか",
    RobotSpeechToken.REASSURE: "ゆっくりで大丈夫です",
    RobotSpeechToken.SAY_WAITING: "待ちます",
    RobotSpeechToken.SAY_RELEASING: "離します",
    RobotSpeechToken.ASK_CONFIRMATION: "取れましたか",
}

HUMAN_SPEECH_TEXT = {
    HumanSpeechEvent.SILENCE: "",
    HumanSpeechEvent.READY: "準備できました",
    HumanSpeechEvent.WAIT: "少し待ってください",
    HumanSpeechEvent.CLOSER: "もう少し近くへ",
    HumanSpeechEvent.SLOWER: "ゆっくりお願いします",
    HumanSpeechEvent.GOT_IT: "受け取りました",
    HumanSpeechEvent.TOO_CLOSE: "近すぎます",
    HumanSpeechEvent.CONFUSED: "どうすればいいですか",
}


def robot_speech_from_name(name: str) -> RobotSpeechToken:
    """Return a robot speech token from a CLI/config name."""
    normalized = name.strip().upper()
    if normalized == "NONE":
        normalized = "SILENCE"
    return RobotSpeechToken[normalized]


def robot_speech_from_scalar(value: float, tokens: Iterable[RobotSpeechToken] | None = None) -> RobotSpeechToken:
    """Map a scalar in [-1, 1] to a discrete robot speech token.

    This is useful for PPO/SAC because the motor action can stay as a continuous
    Box action while the final scalar is binned into a token.
    """
    available = list(tokens or RobotSpeechToken)
    if not available:
        return RobotSpeechToken.SILENCE
    clipped = max(-1.0, min(1.0, float(value)))
    index = int(round((clipped + 1.0) * 0.5 * (len(available) - 1)))
    return available[index]


def robot_speech_to_scalar(token: RobotSpeechToken) -> float:
    """Map a robot speech token to the center of its scalar action bin."""
    tokens = list(RobotSpeechToken)
    if len(tokens) == 1:
        return 0.0
    index = tokens.index(RobotSpeechToken(token))
    return -1.0 + 2.0 * index / (len(tokens) - 1)


def speech_text(token: RobotSpeechToken | HumanSpeechEvent) -> str:
    """Return display text for a robot or human speech token."""
    if isinstance(token, RobotSpeechToken):
        return ROBOT_SPEECH_TEXT[token]
    return HUMAN_SPEECH_TEXT[token]
