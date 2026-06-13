from __future__ import annotations

import unittest
import math
from pathlib import Path

try:
    import mujoco
    import torch

    from allostatic_handover.mjlab_tasks.mdp import (
        HandoverPhase,
        robot_carry_to_hand_reward,
        speech_penalty,
    )
    from allostatic_handover.mjlab_tasks.env_cfg import (
        FULL_HRGYM_OBJECT_Z,
        FULL_TABLE_TOP_Z,
        FULL_YAM_HUMAN_BASE_POS_OFFSET,
        FULL_YAM_OBJECT_FIXED_XY,
        FULL_YAM_OBJECT_INIT_POS,
        FULL_YAM_OBJECT_INIT_ROT,
        FULL_YAM_OBJECT_X_RANGE,
        FULL_YAM_OBJECT_Y_RANGE,
        FULL_YAM_OBJECT_YAW_RANGE,
        FULL_YAM_OBJECT_Z,
        FULL_YAM_GRASPED_START_PEDESTAL_HALF_SIZE,
        FULL_YAM_GRASPED_START_PEDESTAL_POS,
        FULL_YAM_GRASPED_START_ROOT_POS,
        FULL_YAM_HANDOFF_OBJECT_OFFSET,
        FULL_YAM_MIN_GRASP_SITE_Z,
        FULL_YAM_PEDESTAL_HALF_SIZE,
        FULL_YAM_PEDESTAL_POS,
        FULL_YAM_ROOT_POS,
        FULL_YAM_TABLE_SURFACE_ROOT_Z,
        HRGYM_REFERENCE_ROBOT_HAND_POS,
        allostatic_handover_full_allostatic_belief_yam_env_cfg,
        allostatic_handover_full_grasped_start_yam_env_cfg,
        allostatic_handover_full_speech_penalty_yam_env_cfg,
        allostatic_handover_full_task_only_grasped_start_yam_env_cfg,
        allostatic_handover_full_task_only_speech_yam_env_cfg,
        allostatic_handover_full_task_only_yam_env_cfg,
        allostatic_handover_full_yam_env_cfg,
    )
    from allostatic_handover.envs.speech_events import (
        RobotSpeechToken,
        robot_speech_to_scalar,
    )
    from allostatic_handover.envs.human_hidden_state import HumanState
    from allostatic_handover.mjlab_tasks.hrgym_assets import (
        DEFAULT_FULL_ANIMATION_NAMES,
        HRGYM_HUMAN_JOINT_NAMES,
        HrgymAnimationLibrary,
        hrgym_human_spec,
    )
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.utils.spec import auto_wrap_fixed_base_mocap
except Exception as exc:  # pragma: no cover - optional Mjlab dependency
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


HRGYM_ASSET_ROOT = Path(
    "/mnt/k_iwamoto/sim_data/Projects/human-robot-gym/"
    "human_robot_gym/models/assets"
)

HRGYM_REFERENCE_PALM_TARGETS = (
    (2.05705173, 0.54213442, 0.82704406),
    (2.11229471, 0.53659719, 0.81276593),
    (1.89957607, 0.54052568, 0.82035956),
    (2.01651823, 0.55205812, 0.82891683),
    (2.06845228, 0.51554131, 0.82618216),
    (2.05122096, -0.10215259, 0.86009281),
    (2.09572680, -0.10698683, 0.85473646),
    (2.00463780, -0.38811371, 0.83652483),
    (1.79875753, 0.60193472, 0.83078301),
)


@unittest.skipIf(IMPORT_ERROR is not None, f"Mjlab terms unavailable: {IMPORT_ERROR}")
@unittest.skipUnless(HRGYM_ASSET_ROOT.exists(), "human-robot-gym assets are unavailable")
class TestMjlabFullHandoverAssets(unittest.TestCase):
    def test_animation_loader_reads_default_hrgym_handover_set(self) -> None:
        library = HrgymAnimationLibrary(
            vendor_root=HRGYM_ASSET_ROOT,
            animation_names=DEFAULT_FULL_ANIMATION_NAMES,
            device="cpu",
        )
        self.assertEqual(len(library), 9)
        first = library.animations[0]
        self.assertEqual(first.joint_pos.shape[1], len(HRGYM_HUMAN_JOINT_NAMES))
        self.assertGreater(first.num_frames, first.keyframes[1])
        self.assertEqual(first.object_holding_hand, "right")

    def test_normalized_human_xml_compiles_as_mjlab_mocap_entity(self) -> None:
        wrapped = auto_wrap_fixed_base_mocap(lambda: hrgym_human_spec(HRGYM_ASSET_ROOT))()
        model = wrapped.compile()
        self.assertEqual(model.nmocap, 1)
        self.assertEqual(model.njnt, len(HRGYM_HUMAN_JOINT_NAMES))

    def test_animation_metadata_preserves_holding_hand(self) -> None:
        library = HrgymAnimationLibrary(
            vendor_root=HRGYM_ASSET_ROOT,
            animation_names=("RobotHumanHandover/0", "RobotHumanHandover/advanced_5"),
            device="cpu",
        )
        self.assertEqual(library.animations[0].object_holding_hand, "right")
        self.assertEqual(library.animations[1].object_holding_hand, "left")

    def test_full_action_space_keeps_motor_plus_speech_shape(self) -> None:
        cfg = allostatic_handover_full_yam_env_cfg()
        self.assertEqual(set(cfg.actions.keys()), {"arm_ik", "gripper", "speech"})
        self.assertEqual(cfg.commands["handover"].entity_name, "manipulation_object")

    def test_full_object_pose_range_is_separated_from_yam_root_inside_workspace(self) -> None:
        cfg = allostatic_handover_full_yam_env_cfg()
        pose_range = cfg.commands["handover"].object_pose_range
        self.assertEqual(pose_range.x, FULL_YAM_OBJECT_X_RANGE)
        self.assertEqual(pose_range.y, FULL_YAM_OBJECT_Y_RANGE)
        self.assertEqual(pose_range.z, (FULL_YAM_OBJECT_Z, FULL_YAM_OBJECT_Z))
        self.assertEqual(pose_range.yaw, FULL_YAM_OBJECT_YAW_RANGE)

        root_xy = torch.tensor(FULL_YAM_ROOT_POS[:2])
        min_xy = torch.tensor((pose_range.x[0], pose_range.y[0]))
        max_xy = torch.tensor((pose_range.x[1], pose_range.y[1]))
        self.assertGreater(float(torch.norm(min_xy - root_xy)), 0.24)
        self.assertGreater(float(torch.norm(max_xy - root_xy)), 0.24)

    def test_full_allostatic_cfg_has_grasp_and_handoff_learning_terms(self) -> None:
        cfg = allostatic_handover_full_yam_env_cfg()
        command = cfg.commands["handover"]
        self.assertEqual(command.reward_variant, "allostatic")
        self.assertFalse(command.pure_task_mode)
        self.assertTrue(command.robot_grasp_latch_enabled)
        self.assertFalse(command.start_with_object_grasped)
        self.assertTrue(command.allow_release_away_from_hand)
        self.assertEqual(command.release_action_threshold, 0.15)
        self.assertEqual(command.handoff_reach_progress_threshold, 0.1)
        self.assertEqual(command.success_threshold, 0.23)
        self.assertEqual(command.handoff_object_offset, FULL_YAM_HANDOFF_OBJECT_OFFSET)

        for reward_name in (
            "robot_grasp_approach",
            "robot_grasp",
            "carry_to_hand",
            "release_at_hand",
            "handoff",
            "speech_penalty",
            "allostatic_load",
            "waiting_cost",
        ):
            self.assertIn(reward_name, cfg.rewards)
        self.assertEqual(
            cfg.rewards["robot_grasp_approach"].params["object_name"],
            "manipulation_object",
        )
        self.assertGreater(cfg.rewards["robot_grasp_approach"].weight, 0.0)
        self.assertLessEqual(cfg.rewards["robot_grasp"].weight, 0.1)

    def test_full_task_only_cfg_removes_speech_and_allostatic_training_terms(self) -> None:
        cfg = allostatic_handover_full_task_only_yam_env_cfg()
        self.assertEqual(set(cfg.actions.keys()), {"arm_ik", "gripper"})
        command = cfg.commands["handover"]
        self.assertEqual(command.reward_variant, "task_only")
        self.assertTrue(command.pure_task_mode)
        self.assertFalse(command.require_readiness_for_reach)
        self.assertEqual(command.readiness_initial, 1.0)
        self.assertEqual(command.readiness_threshold, 0.0)

        for group in cfg.observations.values():
            for term_name in (
                "readiness_belief",
                "load_proxy",
                "speech_context",
                "privileged_human_state",
                "privileged_load",
            ):
                self.assertNotIn(term_name, group.terms)
        for reward_name in ("speech_penalty", "allostatic_load", "waiting_cost"):
            self.assertNotIn(reward_name, cfg.rewards)
        self.assertTrue(command.robot_grasp_latch_enabled)
        self.assertFalse(command.start_with_object_grasped)
        self.assertTrue(command.allow_release_away_from_hand)
        self.assertFalse(command.freeze_hand_target_after_reset)
        self.assertEqual(command.release_action_threshold, 0.15)
        self.assertEqual(command.handoff_reach_progress_threshold, 0.1)
        self.assertEqual(command.success_threshold, 0.23)
        self.assertEqual(command.min_lift_before_handoff, 0.08)
        self.assertEqual(command.pure_task_min_distance_improvement, 0.0)
        self.assertEqual(command.animation_names, ("RobotHumanHandover/0",))
        self.assertEqual(command.human_base_pos_offset, FULL_YAM_HUMAN_BASE_POS_OFFSET)
        self.assertEqual(command.object_pose_range.x, (FULL_YAM_OBJECT_FIXED_XY[0], FULL_YAM_OBJECT_FIXED_XY[0]))
        self.assertEqual(command.object_pose_range.y, (FULL_YAM_OBJECT_FIXED_XY[1], FULL_YAM_OBJECT_FIXED_XY[1]))
        self.assertEqual(command.object_pose_range.yaw, FULL_YAM_OBJECT_YAW_RANGE)
        self.assertEqual(
            cfg.scene.entities["manipulation_object"].init_state.pos,
            FULL_YAM_OBJECT_INIT_POS,
        )
        self.assertEqual(
            cfg.scene.entities["manipulation_object"].init_state.rot,
            FULL_YAM_OBJECT_INIT_ROT,
        )
        self.assertLess(command.ready_loop_amplitude_scale, command.hesitant_loop_amplitude_scale)
        self.assertEqual(cfg.actions["arm_ik"].delta_pos_scale, 0.10)
        self.assertEqual(cfg.actions["arm_ik"].max_dq, 0.22)
        self.assertIn("robot_grasp", cfg.rewards)
        self.assertIn("handoff", cfg.rewards)
        self.assertIn("time_penalty", cfg.rewards)
        self.assertEqual(cfg.rewards["robot_grasp_approach"].weight, 1.2)
        self.assertEqual(cfg.rewards["robot_grasp"].weight, 0.15)
        self.assertEqual(cfg.rewards["carry_to_hand"].weight, 0.25)
        self.assertEqual(cfg.rewards["action_rate_l2"].weight, -0.002)
        self.assertEqual(cfg.rewards["time_penalty"].weight, -0.02)

    def test_full_task_only_speech_cfg_keeps_speech_and_hides_human_state(self) -> None:
        cfg = allostatic_handover_full_task_only_speech_yam_env_cfg()
        self.assertEqual(set(cfg.actions.keys()), {"arm_ik", "gripper", "speech"})
        command = cfg.commands["handover"]
        self.assertEqual(command.reward_variant, "task_only")
        self.assertFalse(command.pure_task_mode)
        self.assertTrue(command.require_readiness_for_reach)
        self.assertTrue(command.require_readiness_for_animation_start)
        self.assertLess(command.readiness_initial, command.readiness_threshold)
        self.assertGreaterEqual(
            command.readiness_initial + command.announce_effect,
            command.readiness_threshold,
        )
        self.assertEqual(command.overload_threshold, 7.0)
        self.assertEqual(command.withdrawal_threshold, 9.0)
        self.assertEqual(command.animation_names, ("RobotHumanHandover/0",))

        actor_terms = cfg.observations["actor"].terms
        self.assertIn("speech_context", actor_terms)
        for group in cfg.observations.values():
            for hidden_term in (
                "readiness_belief",
                "load_proxy",
                "privileged_human_state",
                "privileged_load",
            ):
                self.assertNotIn(hidden_term, group.terms)
        for reward_name in ("speech_penalty", "allostatic_load", "waiting_cost"):
            self.assertNotIn(reward_name, cfg.rewards)
        self.assertEqual(cfg.rewards["robot_grasp_approach"].weight, 1.2)
        self.assertEqual(cfg.rewards["handoff"].weight, 25.0)

    def test_full_speech_penalty_cfg_uses_hidden_accumulated_speech_load(self) -> None:
        cfg = allostatic_handover_full_speech_penalty_yam_env_cfg()
        self.assertEqual(set(cfg.actions.keys()), {"arm_ik", "gripper", "speech"})
        command = cfg.commands["handover"]
        self.assertEqual(command.reward_variant, "speech_penalty")
        self.assertFalse(command.pure_task_mode)
        self.assertTrue(command.require_readiness_for_reach)
        self.assertTrue(command.require_readiness_for_animation_start)
        self.assertEqual(command.speech_penalty_load_threshold, 0.8)
        self.assertEqual(command.speech_penalty_exp_scale, 0.5)
        self.assertEqual(command.speech_penalty_max_excess, 4.0)
        self.assertEqual(command.overload_threshold, 7.0)
        self.assertEqual(command.withdrawal_threshold, 9.0)
        actor_terms = cfg.observations["actor"].terms
        self.assertIn("speech_context", actor_terms)
        for group in cfg.observations.values():
            for hidden_term in (
                "readiness_belief",
                "load_proxy",
                "privileged_human_state",
                "privileged_load",
            ):
                self.assertNotIn(hidden_term, group.terms)
        self.assertIn("speech_penalty", cfg.rewards)
        self.assertEqual(cfg.rewards["speech_penalty"].weight, -0.04)
        self.assertNotIn("allostatic_load", cfg.rewards)
        self.assertNotIn("waiting_cost", cfg.rewards)
        self.assertEqual(cfg.rewards["success"].weight, 30.0)
        self.assertEqual(cfg.rewards["handoff"].weight, 25.0)

    def test_full_speech_penalty_reward_is_smooth_hidden_load_penalty(self) -> None:
        cfg = allostatic_handover_full_speech_penalty_yam_env_cfg(play=True)
        cfg.scene.num_envs = 1
        env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
        try:
            env.reset()
            command = env.command_manager.get_term("handover")
            command.pre_reward_update()
            command.attention_load[:] = command.cfg.speech_penalty_load_threshold - 0.1
            command.turn_taking_load[:] = 0.0
            command._last_update_step = int(env.common_step_counter)
            below = speech_penalty(env, "handover")[0]

            command.attention_load[:] = command.cfg.speech_penalty_load_threshold + 0.2
            command.turn_taking_load[:] = 0.0
            command._last_update_step = int(env.common_step_counter)
            mild = speech_penalty(env, "handover")[0]

            command.attention_load[:] = command.cfg.speech_penalty_load_threshold + 1.0
            command.turn_taking_load[:] = 0.0
            command._last_update_step = int(env.common_step_counter)
            high = speech_penalty(env, "handover")[0]

            self.assertAlmostEqual(float(below), 0.0, delta=1e-6)
            self.assertGreater(float(mild), 0.0)
            self.assertGreater(float(high), float(mild) * 4.0)

            command.cfg.reward_variant = "allostatic"
            command.attention_load[:] = command.cfg.speech_penalty_load_threshold + 0.2
            command.turn_taking_load[:] = 0.0
            command._last_update_step = int(env.common_step_counter)
            allostatic_mild = speech_penalty(env, "handover")[0]
            self.assertAlmostEqual(float(allostatic_mild), float(mild), delta=1e-6)
        finally:
            env.close()

    def test_full_allostatic_belief_cfg_uses_world_model_observations(self) -> None:
        cfg = allostatic_handover_full_allostatic_belief_yam_env_cfg()
        self.assertEqual(set(cfg.actions.keys()), {"arm_ik", "gripper", "speech"})
        command = cfg.commands["handover"]
        self.assertEqual(command.reward_variant, "allostatic")
        self.assertFalse(command.pure_task_mode)
        self.assertTrue(command.require_readiness_for_reach)
        self.assertTrue(command.require_readiness_for_animation_start)
        self.assertEqual(command.speech_penalty_load_threshold, 0.8)
        self.assertEqual(command.speech_penalty_exp_scale, 0.5)
        self.assertEqual(command.speech_penalty_max_excess, 4.0)
        self.assertEqual(command.overload_threshold, 7.0)
        self.assertEqual(command.withdrawal_threshold, 9.0)

        actor_terms = cfg.observations["actor"].terms
        critic_terms = cfg.observations["critic"].terms
        for hidden_term in (
            "readiness_belief",
            "load_proxy",
            "privileged_human_state",
            "privileged_load",
        ):
            self.assertNotIn(hidden_term, actor_terms)
            self.assertNotIn(hidden_term, critic_terms)
        for wm_term in (
            "wm_belief",
            "wm_human_state_probs",
            "wm_readiness_pred",
            "wm_load_pred",
        ):
            self.assertIn(wm_term, actor_terms)
            self.assertIn(wm_term, critic_terms)
        self.assertEqual(cfg.rewards["speech_penalty"].weight, -0.02)
        self.assertEqual(cfg.rewards["allostatic_load"].weight, -0.05)
        self.assertEqual(cfg.rewards["waiting_cost"].weight, -0.10)
        self.assertEqual(cfg.rewards["success"].weight, 30.0)
        self.assertEqual(cfg.rewards["handoff"].weight, 25.0)

    def test_full_task_only_speech_cue_starts_human_animation(self) -> None:
        cfg = allostatic_handover_full_task_only_speech_yam_env_cfg(play=True)
        cfg.scene.num_envs = 1
        env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
        try:
            env.reset()
            command = env.command_manager.get_term("handover")
            command._last_update_step = -1
            command.pre_reward_update()
            self.assertLess(float(command.human_readiness[0]), command.cfg.readiness_threshold)
            self.assertEqual(float(command._classic_animation_frame[0]), 0.0)
            self.assertEqual(int(command.human_state_id[0]), int(HumanState.HESITANT))

            silence = robot_speech_to_scalar(RobotSpeechToken.SILENCE)
            silent_action = torch.tensor([[0.0, 0.0, 0.0, 0.0, silence]], device=env.device)
            env.action_manager.process_action(silent_action)
            command._last_update_step = -1
            command.pre_reward_update()
            self.assertEqual(float(command._classic_animation_frame[0]), 0.0)

            announce = robot_speech_to_scalar(RobotSpeechToken.ANNOUNCE_HANDOVER)
            cue_action = torch.tensor([[0.0, 0.0, 0.0, 0.0, announce]], device=env.device)
            env.action_manager.process_action(cue_action)
            command._last_update_step = -1
            command.pre_reward_update()
            self.assertGreaterEqual(
                float(command.human_readiness[0]),
                command.cfg.readiness_threshold,
            )
            self.assertGreater(float(command._classic_animation_frame[0]), 0.0)
            self.assertEqual(int(command.human_state_id[0]), int(HumanState.READY))
        finally:
            env.close()

    def test_full_task_only_grasped_start_keeps_old_curriculum_shortcut(self) -> None:
        cfg = allostatic_handover_full_task_only_grasped_start_yam_env_cfg()
        command = cfg.commands["handover"]
        self.assertTrue(command.start_with_object_grasped)
        self.assertEqual(
            cfg.scene.entities["robot"].init_state.pos,
            FULL_YAM_GRASPED_START_ROOT_POS,
        )
        self.assertEqual(
            cfg.scene.entities["yam_pedestal"].init_state.pos,
            FULL_YAM_GRASPED_START_PEDESTAL_POS,
        )

    def test_full_allostatic_grasped_start_uses_current_tabletop_layout(self) -> None:
        cfg = allostatic_handover_full_grasped_start_yam_env_cfg()
        command = cfg.commands["handover"]
        self.assertTrue(command.start_with_object_grasped)
        self.assertFalse(command.allow_release_away_from_hand)
        self.assertEqual(command.reward_variant, "allostatic")
        self.assertIn("speech", cfg.actions)
        self.assertEqual(cfg.scene.entities["robot"].init_state.pos, FULL_YAM_ROOT_POS)
        self.assertEqual(
            cfg.scene.entities["yam_pedestal"].init_state.pos,
            FULL_YAM_PEDESTAL_POS,
        )
        self.assertGreaterEqual(command.readiness_initial + command.announce_effect, 0.65)
        self.assertGreaterEqual(command.readiness_hold_steps, 100)
        self.assertEqual(command.release_action_threshold, 0.0)
        self.assertLess(cfg.rewards["handover"].weight, 0.5)
        self.assertGreaterEqual(cfg.rewards["success"].weight, 40.0)
        self.assertGreaterEqual(cfg.rewards["handoff"].weight, 20.0)
        self.assertIn("release_intent_at_hand", cfg.rewards)
        self.assertGreater(cfg.rewards["release_intent_at_hand"].weight, 0.0)
        self.assertEqual(cfg.rewards["speech_penalty"].weight, -0.02)
        self.assertIn("time_penalty", cfg.rewards)

    def test_registered_full_task_only_root_is_not_contaminated_by_grasped_start(self) -> None:
        import allostatic_handover.mjlab_tasks  # noqa: F401
        from mjlab.tasks.registry import load_env_cfg

        full = load_env_cfg("Mjlab-Allostatic-Handover-Full", play=True)
        full_grasped = load_env_cfg(
            "Mjlab-Allostatic-Handover-Full-GraspedStart",
            play=True,
        )
        task_only = load_env_cfg("Mjlab-Allostatic-Handover-Full-TaskOnly", play=True)
        grasped = load_env_cfg(
            "Mjlab-Allostatic-Handover-Full-TaskOnly-GraspedStart",
            play=True,
        )

        self.assertEqual(full.scene.entities["robot"].init_state.pos, FULL_YAM_ROOT_POS)
        self.assertEqual(full_grasped.scene.entities["robot"].init_state.pos, FULL_YAM_ROOT_POS)
        self.assertFalse(full.commands["handover"].start_with_object_grasped)
        self.assertTrue(full_grasped.commands["handover"].start_with_object_grasped)
        self.assertEqual(task_only.scene.entities["robot"].init_state.pos, FULL_YAM_ROOT_POS)
        self.assertEqual(
            grasped.scene.entities["robot"].init_state.pos,
            FULL_YAM_GRASPED_START_ROOT_POS,
        )
        self.assertNotEqual(
            task_only.scene.entities["robot"].init_state.pos,
            grasped.scene.entities["robot"].init_state.pos,
        )

    def test_full_scene_places_human_upright_and_yam_on_pedestal(self) -> None:
        cfg = allostatic_handover_full_yam_env_cfg(play=True)
        cfg.scene.num_envs = 1
        env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
        try:
            env.reset()

            human = env.scene.entities["human"]
            head_id = human.site_names.index("Head")
            toe_ids = [human.site_names.index("L_Toe"), human.site_names.index("R_Toe")]
            head_z = float(human.data.site_pos_w[0, head_id, 2])
            toe_z = min(float(human.data.site_pos_w[0, idx, 2]) for idx in toe_ids)
            self.assertGreater(head_z, toe_z + 0.8)

            robot = env.scene.entities["robot"]
            robot_root = robot.data.root_link_pos_w[0]
            self.assertAlmostEqual(
                float(robot_root[2]),
                FULL_YAM_TABLE_SURFACE_ROOT_Z,
                delta=0.02,
            )
            self.assertAlmostEqual(float(robot_root[2]), FULL_TABLE_TOP_Z, delta=0.02)
            self.assertTrue(
                torch.allclose(
                    robot_root,
                    torch.tensor(FULL_YAM_ROOT_POS, dtype=robot_root.dtype),
                    atol=2e-2,
                ),
                f"robot_root={robot_root}",
            )

            robot_quat = robot.data.root_link_quat_w[0]
            yaw = self._yaw_from_wxyz(robot_quat)
            self.assertAlmostEqual(yaw, 0.0, delta=0.08)

            grasp_id = robot.site_names.index("grasp_site")
            grasp_pos = robot.data.site_pos_w[0, grasp_id]
            self.assertGreater(float(grasp_pos[2]), FULL_TABLE_TOP_Z)
            self.assertLess(float(torch.norm(grasp_pos[:2] - robot_root[:2])), 0.5)

            self.assertIn("yam_pedestal", env.scene.entities)
            pedestal = env.scene.entities["yam_pedestal"]
            pedestal_root_z = float(pedestal.data.root_link_pos_w[0, 2])
            self.assertAlmostEqual(
                pedestal_root_z,
                FULL_YAM_PEDESTAL_POS[2],
                delta=0.02,
            )
            self.assertAlmostEqual(
                pedestal_root_z + FULL_YAM_PEDESTAL_HALF_SIZE[2],
                FULL_TABLE_TOP_Z,
                delta=0.02,
            )

            obj = env.scene.entities["manipulation_object"]
            self.assertAlmostEqual(
                float(obj.data.root_link_pos_w[0, 2]),
                FULL_YAM_OBJECT_Z,
                delta=1e-4,
            )
            obj_xy = obj.data.root_link_pos_w[0, :2]
            self.assertGreater(float(torch.norm(obj_xy - robot_root[:2])), 0.24)
            self.assertLess(float(torch.norm(obj_xy - grasp_pos[:2])), 0.22)
        finally:
            env.close()

    def test_full_task_only_table_start_does_not_begin_with_object_grasped(self) -> None:
        cfg = allostatic_handover_full_task_only_yam_env_cfg(play=True)
        cfg.scene.num_envs = 1
        env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
        try:
            env.reset()
            command = env.command_manager.get_term("handover")
            command.pre_reward_update()
            robot = env.scene.entities["robot"]
            obj = env.scene.entities["manipulation_object"]
            grasp_id = robot.site_names.index("grasp_site")
            grasp_pos = robot.data.site_pos_w[0, grasp_id]
            obj_pos = obj.data.root_link_pos_w[0]

            self.assertFalse(bool(command.robot_object_grasped[0]))
            self.assertGreater(float(torch.norm(grasp_pos - obj_pos)), 0.05)
            self.assertAlmostEqual(float(obj_pos[2]), FULL_YAM_OBJECT_Z, delta=1e-4)
            obj_quat = obj.data.root_link_quat_w[0]
            expected_quat = torch.tensor(
                FULL_YAM_OBJECT_INIT_ROT,
                dtype=obj_quat.dtype,
                device=obj_quat.device,
            )
            quat_alignment = torch.abs(torch.dot(obj_quat, expected_quat))
            self.assertAlmostEqual(float(quat_alignment), 1.0, delta=1e-4)
        finally:
            env.close()

    def test_full_task_only_object_stays_on_table_after_settling(self) -> None:
        cfg = allostatic_handover_full_task_only_yam_env_cfg(play=True)
        cfg.scene.num_envs = 1
        env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
        try:
            env.reset()
            obj = env.scene.entities["manipulation_object"]
            action = torch.zeros((1, env.action_manager.total_action_dim), device=env.device)
            for _ in range(150):
                env.step(action)
            obj_pos = obj.data.root_link_pos_w[0]
            self.assertGreater(float(obj_pos[2]), FULL_TABLE_TOP_Z - 0.02)
            self.assertLessEqual(float(obj_pos[0]), 0.75)
            self.assertGreaterEqual(float(obj_pos[0]), -0.75)
            self.assertLessEqual(float(obj_pos[1]), 1.0)
            self.assertGreaterEqual(float(obj_pos[1]), -1.0)
        finally:
            env.close()

    def test_full_task_only_table_and_yam_collision_masks_are_enabled(self) -> None:
        cfg = allostatic_handover_full_task_only_yam_env_cfg(play=True)
        cfg.scene.num_envs = 1
        env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
        try:
            env.reset()
            model = env.sim.mj_model

            def geom_id(name: str) -> int:
                gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                self.assertGreaterEqual(gid, 0, name)
                return gid

            table_id = geom_id("table/table_collision")
            pedestal_id = geom_id("yam_pedestal/yam_pedestal_collision")
            yam_geom_ids = [
                geom_id("robot/base_collision"),
                geom_id("robot/link2_1_collision"),
                geom_id("robot/link3_1_collision"),
                geom_id("robot/link4_1_collision"),
                geom_id("robot/link5_1_collision"),
                geom_id("robot/link6_1_collision"),
            ]

            for gid in (table_id, pedestal_id, *yam_geom_ids):
                self.assertGreater(int(model.geom_contype[gid]), 0)
                self.assertGreater(int(model.geom_conaffinity[gid]), 0)

            for surface_id in (table_id, pedestal_id):
                for yam_id in yam_geom_ids:
                    can_collide = (
                        int(model.geom_contype[surface_id])
                        & int(model.geom_conaffinity[yam_id])
                    ) or (
                        int(model.geom_contype[yam_id])
                        & int(model.geom_conaffinity[surface_id])
                    )
                    self.assertNotEqual(can_collide, 0)
        finally:
            env.close()

    def test_full_task_only_arm_ik_target_is_clamped_above_table(self) -> None:
        cfg = allostatic_handover_full_task_only_yam_env_cfg(play=True)
        cfg.scene.num_envs = 1
        env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
        try:
            env.reset()
            term = env.action_manager.get_term("arm_ik")
            term.process_actions(torch.tensor([[0.0, 0.0, -10.0]], device=env.device))
            self.assertGreaterEqual(
                float(term._desired_pos[0, 2]),
                FULL_YAM_MIN_GRASP_SITE_Z - 1e-6,
            )
        finally:
            env.close()

    def test_full_task_only_gripper_geoms_do_not_penetrate_table_under_down_action(
        self,
    ) -> None:
        cfg = allostatic_handover_full_task_only_yam_env_cfg(play=True)
        cfg.scene.num_envs = 1
        env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
        try:
            env.reset()
            model = env.sim.mj_model
            gripper_geom_ids = []
            for geom_id in range(model.ngeom):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
                body_name = (
                    mujoco.mj_id2name(
                        model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        int(model.geom_bodyid[geom_id]),
                    )
                    or ""
                )
                if name.startswith("robot/") and any(
                    token in name for token in ("lf_", "rf_", "link6")
                ):
                    gripper_geom_ids.append(geom_id)
                elif body_name.startswith("robot/") and any(
                    token in body_name
                    for token in ("link_left_finger", "link_right_finger", "link_6")
                ):
                    gripper_geom_ids.append(geom_id)
            self.assertGreater(len(gripper_geom_ids), 0)

            action = torch.tensor([[0.0, 0.0, -1.0, 0.0]], device=env.device)
            for _ in range(80):
                env.step(action)
            env.sim.forward()
            env.scene.update(0.0)
            min_gripper_z = float(env.sim.data.geom_xpos[0, gripper_geom_ids, 2].min())
            self.assertGreater(min_gripper_z, FULL_TABLE_TOP_Z)
        finally:
            env.close()

    def test_full_task_only_grasped_start_begins_with_object_at_grasp_site(self) -> None:
        cfg = allostatic_handover_full_task_only_grasped_start_yam_env_cfg(play=True)
        cfg.scene.num_envs = 1
        env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
        try:
            env.reset()
            command = env.command_manager.get_term("handover")
            command.pre_reward_update()
            robot = env.scene.entities["robot"]
            obj = env.scene.entities["manipulation_object"]
            grasp_id = robot.site_names.index("grasp_site")
            grasp_pos = robot.data.site_pos_w[0, grasp_id]
            obj_pos = obj.data.root_link_pos_w[0]

            self.assertTrue(bool(command.robot_object_grasped[0]))
            self.assertLess(
                float(torch.norm(grasp_pos - obj_pos)),
                command.cfg.robot_grasp_distance_threshold,
            )
            self.assertAlmostEqual(
                float(robot.data.root_link_pos_w[0, 2]),
                FULL_YAM_GRASPED_START_ROOT_POS[2],
                delta=0.02,
            )
            self.assertAlmostEqual(
                FULL_YAM_GRASPED_START_PEDESTAL_POS[2]
                + FULL_YAM_GRASPED_START_PEDESTAL_HALF_SIZE[2],
                FULL_YAM_GRASPED_START_ROOT_POS[2],
                delta=1e-6,
            )
        finally:
            env.close()

    def test_readiness_reduces_hrgym_loop_amplitude_scale(self) -> None:
        cfg = allostatic_handover_full_task_only_yam_env_cfg(play=True)
        cfg.scene.num_envs = 1
        env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
        try:
            env.reset()
            command = env.command_manager.get_term("handover")

            command.human_readiness[:] = 0.0
            hesitant_scale = command._readiness_loop_amplitude_scale()[0]
            command.human_readiness[:] = 1.0
            ready_scale = command._readiness_loop_amplitude_scale()[0]

            self.assertGreater(float(hesitant_scale), float(ready_scale))
            self.assertAlmostEqual(
                float(ready_scale),
                command.cfg.ready_loop_amplitude_scale,
                delta=1e-6,
            )
        finally:
            env.close()

    def test_full_task_only_grasped_start_env_has_four_actions_and_ready_human(self) -> None:
        cfg = allostatic_handover_full_task_only_grasped_start_yam_env_cfg(play=True)
        cfg.scene.num_envs = 1
        env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
        try:
            env.reset()
            self.assertEqual(env.action_space.shape[-1], 4)
            command = env.command_manager.get_term("handover")
            self.assertAlmostEqual(float(command.human_readiness[0]), 1.0, delta=1e-6)
            self.assertAlmostEqual(float(command.allostatic_load_total[0]), 0.0, delta=1e-6)
            self.assertTrue(bool(command.robot_object_grasped[0]))
            command._last_update_step = -1
            command.pre_reward_update()
            env.sim.forward()
            env.scene.update(0.0)
            grasp_id = command._grasp_site_id
            ee_pos = env.scene.entities["robot"].data.site_pos_w[0, grasp_id]
            obj_pos = env.scene.entities["manipulation_object"].data.root_link_pos_w[0]
            self.assertLess(
                float(torch.norm(obj_pos - ee_pos)),
                command.cfg.robot_grasp_distance_threshold,
            )
            frozen_target = command.hand_pos[0].clone()
            command.animation_frame[:] = 500.0
            command._classic_animation_frame[:] = 500.0
            command._last_update_step = -1
            command.pre_reward_update()
            self.assertFalse(torch.allclose(command.hand_pos[0], frozen_target, atol=1e-3))

            release_gripper = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=env.device)
            env.action_manager.process_action(release_gripper)
            command._last_update_step = -1
            command.pre_reward_update()
            self.assertFalse(bool(command.robot_object_grasped[0]))
        finally:
            env.close()

    def test_full_task_only_keeps_human_off_table_and_reaches_into_workspace(self) -> None:
        cfg = allostatic_handover_full_task_only_yam_env_cfg(play=True)
        cfg.scene.num_envs = 1
        env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
        try:
            env.reset()
            command = env.command_manager.get_term("handover")
            human = env.scene.entities["human"]
            toe_ids = [human.site_names.index("L_Toe"), human.site_names.index("R_Toe")]

            command.animation_id[:] = 0
            key0, key1, _ = command._current_keyframe_tensors()
            frame = key0 + 0.4 * (key1 - key0)
            command.phase[:] = int(HandoverPhase.REACH_OUT)
            command.animation_frame[:] = frame
            command._classic_animation_frame[:] = frame
            command._delayed_animation_frames[:] = 0.0
            command._write_human_animation_state()
            env.sim.forward()
            env.scene.update(0.0)

            hand = command._read_palm_target()[0]
            robot = env.scene.entities["robot"]
            grasp_id = command._grasp_site_id
            grasp = robot.data.site_pos_w[0, grasp_id]
            toes = torch.stack([human.data.site_pos_w[0, idx] for idx in toe_ids])
            table_clearance = torch.norm(toes[:, :2] - grasp[:2], dim=-1).min()
            self.assertGreater(float(table_clearance), 0.25)
            self.assertLess(float(torch.norm(hand - grasp)), 0.60)
        finally:
            env.close()

    def test_full_task_only_latches_robot_grasp_near_object(self) -> None:
        cfg = allostatic_handover_full_task_only_yam_env_cfg(play=True)
        cfg.scene.num_envs = 1
        env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
        try:
            env.reset()
            command = env.command_manager.get_term("handover")
            command._last_update_step = -1
            command.pre_reward_update()
            self.assertEqual(int(command.phase[0]), int(HandoverPhase.REACH_OUT))
            grasp_id = command._grasp_site_id
            ee_pos = env.scene.entities["robot"].data.site_pos_w[:, grasp_id, :]
            command._write_entity_pose(command.object, ee_pos, torch.tensor([0], device=env.device))
            env.sim.forward()
            env.scene.update(0.0)

            close_gripper = torch.tensor([[0.0, 0.0, 0.0, -1.0]], device=env.device)
            env.action_manager.process_action(close_gripper)
            command._last_update_step = -1
            command.pre_reward_update()

            self.assertTrue(bool(command.robot_object_grasped[0]))
            obj_pos = command.object.data.root_link_pos_w[0]
            self.assertTrue(torch.allclose(obj_pos, ee_pos[0], atol=1e-5), obj_pos)
        finally:
            env.close()

    def test_full_task_only_carry_reward_stays_positive_and_prefers_near_holding(self) -> None:
        cfg = allostatic_handover_full_task_only_yam_env_cfg(play=True)
        cfg.scene.num_envs = 1
        env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
        try:
            env.reset()
            command = env.command_manager.get_term("handover")
            command._last_update_step = -1
            command.pre_reward_update()
            command.robot_object_grasped[:] = True
            radius = command.cfg.success_threshold

            command.palm_distance[:] = 2.0 * radius
            far_reward = robot_carry_to_hand_reward(env, "handover", std=0.5)[0]

            command.palm_distance[:] = 0.5 * radius
            near_reward = robot_carry_to_hand_reward(env, "handover", std=0.5)[0]

            self.assertGreaterEqual(float(far_reward), 0.0)
            self.assertGreater(float(near_reward), 0.0)
            self.assertGreater(float(near_reward), float(far_reward))

            command.robot_reached_hand[:] = True
            reached_reward = robot_carry_to_hand_reward(env, "handover", std=0.5)[0]
            self.assertEqual(float(reached_reward), 0.0)
        finally:
            env.close()

    def test_full_task_only_handoff_after_grasp_and_release_at_hand(self) -> None:
        cfg = allostatic_handover_full_task_only_yam_env_cfg(play=True)
        cfg.scene.num_envs = 1
        env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
        try:
            env.reset()
            command = env.command_manager.get_term("handover")
            key0, key1, _ = command._current_keyframe_tensors()
            frame = key0 + 0.4 * (key1 - key0)
            command.phase[:] = int(HandoverPhase.REACH_OUT)
            command.animation_frame[:] = frame
            command._classic_animation_frame[:] = frame
            command._delayed_animation_frames[:] = 0.0
            command.human_readiness[:] = 1.0
            command.robot_object_grasped[:] = True
            command._write_human_animation_state()
            env.sim.forward()
            env.scene.update(0.0)
            command._last_update_step = -1
            command.pre_reward_update()
            hand_pos = command.hand_pos
            grasp_id = command._grasp_site_id
            ee_pos = env.scene.entities["robot"].data.site_pos_w[:, grasp_id, :]
            command._robot_grasp_object_offset = hand_pos[0] - ee_pos[0]
            command._write_entity_pose(command.object, hand_pos, torch.tensor([0], device=env.device))

            release_gripper = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=env.device)
            env.action_manager.process_action(release_gripper)
            command._last_update_step = -1
            command.pre_reward_update()

            self.assertTrue(bool(command.object_attached[0]))
            self.assertFalse(bool(command.robot_object_grasped[0]))
            self.assertEqual(int(command.phase[0]), int(HandoverPhase.COMPLETE))
            self.assertEqual(float(command.episode_success[0]), 1.0)
        finally:
            env.close()

    def test_full_palm_targets_match_translated_hrgym_reference_targets(self) -> None:
        cfg = allostatic_handover_full_yam_env_cfg(play=True)
        cfg.scene.num_envs = 1
        env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
        try:
            env.reset()
            command = env.command_manager.get_term("handover")
            self.assertEqual(len(command.animation_library), len(HRGYM_REFERENCE_PALM_TARGETS))

            for animation_id, expected in enumerate(HRGYM_REFERENCE_PALM_TARGETS):
                command.animation_id[:] = animation_id
                command.animation_frame[:] = 0.0
                command._classic_animation_frame[:] = 0.0
                command._delayed_animation_frames[:] = 0.0
                command._write_human_animation_state()
                env.sim.forward()
                env.scene.update(0.0)

                target = command._read_palm_target()[0]
                expected_tensor = torch.tensor(expected, dtype=target.dtype)
                expected_tensor = expected_tensor + torch.tensor(
                    FULL_YAM_HUMAN_BASE_POS_OFFSET,
                    dtype=target.dtype,
                )
                self.assertTrue(
                    torch.allclose(target, expected_tensor, atol=1e-3),
                    f"animation={animation_id} target={target} expected={expected_tensor}",
                )
        finally:
            env.close()

    def _yaw_from_wxyz(self, quat: torch.Tensor) -> float:
        w, x, y, z = [float(v) for v in quat]
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


if __name__ == "__main__":
    unittest.main()
