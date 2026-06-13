"""Unit tests for allostatic world-model belief utilities."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from allostatic_handover.world_model import (
    BeliefEstimator,
    BeliefModelConfig,
    WorldModelSequenceDataset,
    load_belief_model,
    save_belief_model,
)
from allostatic_handover.world_model.dataset import compute_normalization
from allostatic_handover.world_model.training import train_belief_world_model
from allostatic_handover.dreamerv3_exact.dataset import (
    DreamerBatchConfig,
    OfflineWorldModelBatchStream,
)
from allostatic_handover.dreamerv3_exact.dependencies import (
    check_dreamerv3_dependencies,
)


class WorldModelPipelineTest(unittest.TestCase):
    def _arrays(self, time: int = 12, num_envs: int = 3) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(7)
        public_obs = rng.normal(size=(time, num_envs, 31)).astype(np.float32)
        action = rng.uniform(-1.0, 1.0, size=(time, num_envs, 5)).astype(np.float32)
        human_state_id = (action[..., 4] > -0.2).astype(np.int64)
        human_readiness = np.clip(0.25 + 0.5 * human_state_id + 0.05 * rng.normal(size=(time, num_envs)), 0.0, 1.0).astype(np.float32)
        load = np.clip(np.abs(action[..., 4]) * 0.25, 0.0, 2.0).astype(np.float32)
        return {
            "public_obs": public_obs,
            "action": action,
            "reward": rng.normal(size=(time, num_envs)).astype(np.float32),
            "done": np.zeros((time, num_envs), dtype=np.float32),
            "human_state_id": human_state_id,
            "human_readiness": human_readiness,
            "allostatic_load_total": load,
            "phase": np.zeros((time, num_envs), dtype=np.int64),
            "reach_progress": rng.uniform(0.0, 1.0, size=(time, num_envs)).astype(np.float32),
        }

    def test_sequence_dataset_returns_action_and_hidden_labels(self) -> None:
        dataset = WorldModelSequenceDataset(self._arrays(), seq_len=4, stride=4)
        sample = dataset[0]
        self.assertEqual(sample["public_obs"].shape, (4, 31))
        self.assertEqual(sample["action"].shape, (4, 5))
        self.assertEqual(sample["human_state_id"].shape, (4,))
        self.assertIn("allostatic_load_total", sample)

    def test_belief_model_one_batch_outputs_auxiliary_heads(self) -> None:
        arrays = self._arrays()
        normalization = compute_normalization(arrays)
        model = BeliefEstimator(BeliefModelConfig(), normalization)
        batch = WorldModelSequenceDataset(arrays, seq_len=4)[0]
        outputs = model.forward_sequence(
            batch["public_obs"].unsqueeze(0),
            batch["action"].unsqueeze(0),
            batch["done"].unsqueeze(0),
        )
        self.assertEqual(outputs["belief"].shape[-1], 16)
        self.assertEqual(outputs["human_state_probs"].shape[-1], 6)
        self.assertEqual(outputs["readiness"].shape[-1], 1)
        self.assertEqual(outputs["load"].shape[-1], 1)

    def test_belief_model_save_load_roundtrip(self) -> None:
        arrays = self._arrays()
        normalization = compute_normalization(arrays)
        model = BeliefEstimator(BeliefModelConfig(), normalization)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "belief_distill.pt"
            save_belief_model(path, model, normalization, metrics={"human_state_acc": 0.5})
            loaded, metadata = load_belief_model(path)
        self.assertIsInstance(loaded, BeliefEstimator)
        self.assertEqual(metadata["metrics"]["human_state_acc"], 0.5)

    def test_training_smoke_writes_required_artifacts(self) -> None:
        arrays = self._arrays(time=16, num_envs=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset_path = tmp / "dataset.npz"
            np.savez_compressed(dataset_path, **arrays)
            metrics = train_belief_world_model(
                dataset_path,
                tmp / "wm",
                updates=1,
                batch_size=2,
                seq_len=4,
                device="cpu",
            )
            self.assertIn("human_state_acc", metrics)
            self.assertTrue((tmp / "wm" / "world_model.ckpt").exists())
            self.assertTrue((tmp / "wm" / "belief_distill.pt").exists())
            self.assertTrue((tmp / "wm" / "normalization.json").exists())
            self.assertTrue((tmp / "wm" / "metrics.jsonl").exists())

    def test_exact_dreamer_stream_returns_obs_action_and_ext_labels(self) -> None:
        arrays = self._arrays(time=16, num_envs=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "dataset.npz"
            np.savez_compressed(dataset_path, **arrays)
            stream = OfflineWorldModelBatchStream(
                dataset_path,
                DreamerBatchConfig(batch_size=2, batch_length=5, seed=11),
            )
            batch = next(iter(stream))
        self.assertEqual(batch["public_obs"].shape, (2, 5, 31))
        self.assertEqual(batch["action"].shape, (2, 5, 5))
        self.assertEqual(batch["human_state_id"].shape, (2, 5))
        self.assertEqual(batch["human_readiness"].shape, (2, 5))
        self.assertEqual(batch["allostatic_load_total"].shape, (2, 5))
        obs_keys = {"public_obs", "reward", "is_first", "is_last", "is_terminal"}
        self.assertNotIn("human_state_id", obs_keys)
        self.assertNotIn("human_readiness", obs_keys)
        self.assertNotIn("allostatic_load_total", obs_keys)

    def test_exact_dependency_check_can_report_missing_module(self) -> None:
        status = check_dreamerv3_dependencies(
            dreamerv3_path="/tmp/does-not-matter",
            modules=("math", "definitely_missing_allostatic_test_module"),
        )
        self.assertFalse(status.ok)
        self.assertIn("definitely_missing_allostatic_test_module", status.missing)


if __name__ == "__main__":
    unittest.main()
