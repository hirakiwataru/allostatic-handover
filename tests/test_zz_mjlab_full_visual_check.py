from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
    from PIL import Image

    from allostatic_handover.mjlab_tasks.env_cfg import (
        FULL_TABLE_TOP_Z,
        FULL_YAM_GRASPED_START_ROOT_POS,
        FULL_YAM_PEDESTAL_HALF_SIZE,
        FULL_YAM_ROOT_POS,
        HRGYM_REFERENCE_ROBOT_HAND_POS,
        allostatic_handover_full_task_only_grasped_start_yam_env_cfg,
        allostatic_handover_full_task_only_yam_env_cfg,
        allostatic_handover_full_yam_env_cfg,
    )
    from allostatic_handover.mjlab_tasks.hrgym_assets import DEFAULT_VENDOR_ROOT
    from allostatic_handover.mjlab_tasks.mdp import HandoverPhase
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
except Exception as exc:  # pragma: no cover - optional Mjlab dependency
    IMPORT_ERROR = exc
    DEFAULT_VENDOR_ROOT = Path("__missing_mjlab_vendor_root__")
else:
    IMPORT_ERROR = None


OUTPUT_DIR = Path(
    "/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/"
    "outputs/visual_checks/mjlab_full"
)
TASK_ONLY_OUTPUT_DIR = Path(
    "/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/"
    "outputs/visual_checks/mjlab_full_task_only"
)


@unittest.skipIf(IMPORT_ERROR is not None, f"Mjlab visual check unavailable: {IMPORT_ERROR}")
@unittest.skipUnless(DEFAULT_VENDOR_ROOT.exists(), "Run make copy-hrgym-full-assets first")
class TestZzMjlabFullVisualCheck(unittest.TestCase):
    def test_offscreen_visual_check_writes_nonblank_stage_images(self) -> None:
        os.environ.setdefault("MUJOCO_GL", "egl")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        cfg = allostatic_handover_full_yam_env_cfg(play=True)
        cfg.scene.num_envs = 1
        cfg.scene.env_spacing = 2.4
        env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode="rgb_array")
        try:
            env.reset()
            command = env.command_manager.get_term("handover")
            self._assert_scene_layout(env)
            frames = {
                "reset": self._render(env),
            }

            command.human_readiness[:] = 1.0
            key0, key1, _ = command._current_keyframe_tensors()
            command.phase[:] = int(HandoverPhase.APPROACH)
            command._classic_animation_frame[:] = key0 + 8.0
            command._last_update_step = -1
            command.pre_reward_update()
            env.scene.write_data_to_sim()
            env.sim.forward()
            frames["reach_out"] = self._render(env)

            command.phase[:] = int(HandoverPhase.RETREAT)
            command.object_attached[:] = True
            command._classic_animation_frame[:] = key1 + 20.0
            command._last_update_step = -1
            command.pre_reward_update()
            env.scene.write_data_to_sim()
            env.sim.forward()
            frames["handoff"] = self._render(env)

            for name, frame in frames.items():
                path = OUTPUT_DIR / f"{name}.png"
                Image.fromarray(frame).save(path)
                self._assert_nonblank(frame, path)
                self._assert_foreground_bbox(frame, path)
        finally:
            env.close()

    def test_task_only_visual_check_keeps_human_off_table(self) -> None:
        os.environ.setdefault("MUJOCO_GL", "egl")
        TASK_ONLY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        cfg = allostatic_handover_full_task_only_yam_env_cfg(play=True)
        cfg.scene.num_envs = 1
        cfg.scene.env_spacing = 2.4
        env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode="rgb_array")
        try:
            env.reset()
            command = env.command_manager.get_term("handover")
            self._assert_scene_layout(env)
            frames = {
                "reset": self._render(env),
            }

            command.human_readiness[:] = 1.0
            key0, key1, _ = command._current_keyframe_tensors()
            frame = key0 + 0.4 * (key1 - key0)
            command.phase[:] = int(HandoverPhase.REACH_OUT)
            command.animation_frame[:] = frame
            command._classic_animation_frame[:] = frame
            command._delayed_animation_frames[:] = 0.0
            command._last_update_step = -1
            command.pre_reward_update()
            env.scene.write_data_to_sim()
            env.sim.forward()
            env.scene.update(0.0)
            self._assert_human_reaches_from_outside_table(env, command)
            frames["reach_out"] = self._render(env)

            command.phase[:] = int(HandoverPhase.RETREAT)
            command.object_attached[:] = True
            command._classic_animation_frame[:] = key1 + 20.0
            command._last_update_step = -1
            command.pre_reward_update()
            env.scene.write_data_to_sim()
            env.sim.forward()
            env.scene.update(0.0)
            frames["handoff"] = self._render(env)

            for name, frame in frames.items():
                path = TASK_ONLY_OUTPUT_DIR / f"{name}.png"
                Image.fromarray(frame).save(path)
                self._assert_nonblank(frame, path)
                self._assert_foreground_bbox(frame, path)
        finally:
            env.close()

    def _render(self, env: ManagerBasedRlEnv) -> np.ndarray:
        frame = env.render()
        self.assertIsNotNone(frame)
        assert frame is not None
        if frame.ndim == 4:
            frame = frame[0]
        return np.asarray(frame, dtype=np.uint8)

    def _assert_nonblank(self, frame: np.ndarray, path: Path) -> None:
        self.assertEqual(frame.ndim, 3, str(path))
        self.assertEqual(frame.shape[-1], 3, str(path))
        self.assertGreater(float(frame.std()), 1.0, str(path))
        corner = frame[0, 0].astype(np.int16)
        diff = np.max(np.abs(frame.astype(np.int16) - corner), axis=-1)
        self.assertGreater(float((diff > 5).mean()), 0.01, str(path))

    def _assert_foreground_bbox(self, frame: np.ndarray, path: Path) -> None:
        corner = frame[0, 0].astype(np.int16)
        diff = np.max(np.abs(frame.astype(np.int16) - corner), axis=-1)
        ys, xs = np.where(diff > 5)
        self.assertGreater(len(xs), 0, str(path))
        width = int(xs.max() - xs.min() + 1)
        height = int(ys.max() - ys.min() + 1)
        self.assertGreater(width, 80, str(path))
        self.assertGreater(height, 80, str(path))

    def _assert_scene_layout(self, env: ManagerBasedRlEnv) -> None:
        self.assertIn("yam_pedestal", env.scene.entities)
        for name in ("robot", "table", "human", "manipulation_object"):
            self.assertIn(name, env.scene.entities)

        human = env.scene.entities["human"]
        head_id = human.site_names.index("Head")
        toe_ids = [human.site_names.index("L_Toe"), human.site_names.index("R_Toe")]
        head_z = float(human.data.site_pos_w[0, head_id, 2])
        toe_z = min(float(human.data.site_pos_w[0, idx, 2]) for idx in toe_ids)
        self.assertGreater(head_z, toe_z + 0.8)

        pedestal = env.scene.entities["yam_pedestal"]
        pedestal_root_z = float(pedestal.data.root_link_pos_w[0, 2])
        self.assertAlmostEqual(
            pedestal_root_z + FULL_YAM_PEDESTAL_HALF_SIZE[2],
            FULL_TABLE_TOP_Z,
            delta=0.02,
        )

        robot = env.scene.entities["robot"]
        robot_root = robot.data.root_link_pos_w[0]
        self.assertTrue(
            torch.allclose(
                robot_root,
                torch.tensor(FULL_YAM_ROOT_POS, dtype=robot_root.dtype),
                atol=2e-2,
            ),
            f"robot_root={robot_root}",
        )

        grasp_id = robot.site_names.index("grasp_site")
        grasp_pos = robot.data.site_pos_w[0, grasp_id]
        self.assertGreater(float(grasp_pos[2]), FULL_TABLE_TOP_Z + 0.05)

    def test_grasped_start_visual_layout_keeps_old_checkpoint_workspace(self) -> None:
        cfg = allostatic_handover_full_task_only_grasped_start_yam_env_cfg(play=True)
        cfg.scene.num_envs = 1
        env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode="rgb_array")
        try:
            env.reset()
            robot = env.scene.entities["robot"]
            robot_root = robot.data.root_link_pos_w[0]
            self.assertTrue(
                torch.allclose(
                    robot_root,
                    torch.tensor(FULL_YAM_GRASPED_START_ROOT_POS, dtype=robot_root.dtype),
                    atol=2e-2,
                ),
                f"robot_root={robot_root}",
            )
            grasp_id = robot.site_names.index("grasp_site")
            grasp_pos = robot.data.site_pos_w[0, grasp_id]
            self.assertGreater(float(grasp_pos[2]), FULL_TABLE_TOP_Z + 0.4)
        finally:
            env.close()

    def _assert_human_reaches_from_outside_table(self, env: ManagerBasedRlEnv, command) -> None:
        human = env.scene.entities["human"]
        toe_ids = [human.site_names.index("L_Toe"), human.site_names.index("R_Toe")]
        toes = torch.stack([human.data.site_pos_w[0, idx] for idx in toe_ids])
        self.assertGreater(float(toes[:, 0].min()), 0.75)

        robot = env.scene.entities["robot"]
        grasp_id = robot.site_names.index("grasp_site")
        grasp_pos = robot.data.site_pos_w[0, grasp_id]
        hand_pos = command._read_palm_target()[0]
        self.assertLess(float(torch.norm(hand_pos - grasp_pos)), 0.65)


if __name__ == "__main__":
    unittest.main()
