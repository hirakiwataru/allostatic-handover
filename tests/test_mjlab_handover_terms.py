from __future__ import annotations

import unittest


try:
    import torch

    from allostatic_handover.envs.speech_events import RobotSpeechToken
    from allostatic_handover.mjlab_tasks.mdp.commands import (
        AllostaticHandoverCommandCfg,
        speech_tokens_from_scalar,
    )
except Exception as exc:  # pragma: no cover - optional Mjlab dependency
    torch = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(IMPORT_ERROR is not None, f"Mjlab terms unavailable: {IMPORT_ERROR}")
class TestMjlabHandoverTerms(unittest.TestCase):
    def test_speech_scalar_bins_match_existing_token_order(self) -> None:
        values = torch.tensor([-1.0, -0.66, -0.33, 0.0, 0.33, 0.66, 1.0])
        tokens = speech_tokens_from_scalar(values)
        self.assertEqual(tokens[0].item(), int(RobotSpeechToken.SILENCE))
        self.assertEqual(tokens[1].item(), int(RobotSpeechToken.ANNOUNCE_HANDOVER))
        self.assertEqual(tokens[3].item(), int(RobotSpeechToken.REASSURE))
        self.assertEqual(tokens[-1].item(), int(RobotSpeechToken.ASK_CONFIRMATION))

    def test_reward_variant_defaults_to_allostatic(self) -> None:
        cfg = AllostaticHandoverCommandCfg(resampling_time_range=(1000.0, 1000.0))
        self.assertEqual(cfg.reward_variant, "allostatic")


if __name__ == "__main__":
    unittest.main()
