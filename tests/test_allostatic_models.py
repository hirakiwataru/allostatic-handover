import unittest

import numpy as np

from allostatic_handover.envs.allostatic_load import AllostaticLoadModel, InteractionFeatures
from allostatic_handover.envs.human_hidden_state import HumanHiddenStateMachine, HumanState
from allostatic_handover.envs.mock_handover_env import MockAllostaticHandoverEnv
from allostatic_handover.envs.reward_variants import RewardContext, compute_reward
from allostatic_handover.envs.speech_events import RobotSpeechToken, robot_speech_to_scalar
from allostatic_handover.wrappers.hrgym_training_stack import SpeechActionWrapper


class AllostaticModelsTest(unittest.TestCase):
    def test_repeated_speech_increases_load(self):
        model = AllostaticLoadModel()
        first = model.update(
            InteractionFeatures(
                robot_speech=RobotSpeechToken.ASK_READY,
                previous_robot_speech=RobotSpeechToken.SILENCE,
            )
        )["allostatic_load_total"]
        second = model.update(
            InteractionFeatures(
                robot_speech=RobotSpeechToken.ASK_READY,
                previous_robot_speech=RobotSpeechToken.ASK_READY,
            )
        )["allostatic_load_total"]
        self.assertGreater(second, first)

    def test_reassure_can_recover_hesitation(self):
        fsm = HumanHiddenStateMachine()
        fsm.force_state(HumanState.HESITANT)
        output = fsm.update(
            InteractionFeatures(
                robot_speech=RobotSpeechToken.REASSURE,
                previous_robot_speech=RobotSpeechToken.SILENCE,
            ),
            {"allostatic_load_total": 0.5},
        )
        self.assertEqual(output.state, HumanState.READY)

    def test_allostatic_reward_penalizes_load(self):
        task = 1.0
        reward = compute_reward(
            task,
            "allostatic",
            RewardContext(allostatic_load_total=2.0, robot_speech_count_step=1.0),
        )
        self.assertLess(reward, task)

    def test_speech_action_wrapper_splits_motor_and_speech(self):
        class DummyMotorEnv:
            action_spec = (
                np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
                np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            )
            observation_space = None

            def __init__(self):
                self.last_action = None

            def reset(self):
                return np.zeros(1, dtype=np.float32)

            def step(self, action):
                self.last_action = np.asarray(action, dtype=np.float32)
                return np.zeros(1, dtype=np.float32), 0.0, False, False, {}

        class SpeechEnv:
            def __init__(self):
                self.last_speech = RobotSpeechToken.SILENCE

            def set_robot_speech(self, token):
                self.last_speech = RobotSpeechToken(token)

        motor_env = DummyMotorEnv()
        speech_env = SpeechEnv()
        wrapper = SpeechActionWrapper(motor_env, speech_env=speech_env)
        action = np.array(
            [0.1, -0.2, 0.3, 0.4, robot_speech_to_scalar(RobotSpeechToken.ASK_READY)],
            dtype=np.float32,
        )
        wrapper.step(action)
        np.testing.assert_allclose(motor_env.last_action, action[:4])
        self.assertEqual(speech_env.last_speech, RobotSpeechToken.ASK_READY)
        self.assertEqual(wrapper.action_space.shape, (5,))

    def test_privileged_observation_default_hides_state_and_load(self):
        env = MockAllostaticHandoverEnv(privileged_observation=False)
        obs, _info = env.reset()
        privileged_env = MockAllostaticHandoverEnv(privileged_observation=True)
        privileged_obs, _privileged_info = privileged_env.reset()
        self.assertEqual(obs.shape[0], 13)
        self.assertEqual(privileged_obs.shape[0], 15)

    def test_silence_blocks_contact_when_human_not_ready(self):
        env = MockAllostaticHandoverEnv(privileged_observation=False)
        env.reset()
        env.object_gripped = True
        env.robot_pos = env.human_hand_rest_pos.copy()
        env.object_pos = env.human_hand_rest_pos.copy()
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        action[3] = 1.0
        action[-1] = robot_speech_to_scalar(RobotSpeechToken.SILENCE)
        _obs, _reward, terminated, _truncated, info = env.step(action)
        self.assertFalse(terminated)
        self.assertFalse(info["success"])
        self.assertTrue(info["acceptance_blocked"])
        self.assertLess(info["human_readiness"], info["readiness_threshold"])
        self.assertLess(info["human_reach_progress"], info["readiness_threshold"])

    def test_speech_can_prepare_human_for_contact_without_load_gate(self):
        env = MockAllostaticHandoverEnv(privileged_observation=False)
        env.reset()
        env.object_gripped = True
        env.robot_pos = env.human_hand_extended_pos.copy()
        env.object_pos = env.human_hand_extended_pos.copy()
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        action[3] = 1.0
        action[-1] = robot_speech_to_scalar(RobotSpeechToken.ANNOUNCE_HANDOVER)
        terminated = False
        info = {}
        for _ in range(20):
            _obs, _reward, terminated, _truncated, info = env.step(action)
            if terminated:
                break
        self.assertTrue(terminated)
        self.assertTrue(info["success"])
        self.assertGreaterEqual(info["human_readiness"], info["readiness_threshold"])
        self.assertGreaterEqual(
            info["human_reach_progress"],
            env.readiness_config.min_reach_progress_for_contact,
        )
        self.assertGreater(info["allostatic_load_total"], 0.0)

    def test_readiness_and_reach_progress_decay_with_silence(self):
        env = MockAllostaticHandoverEnv(privileged_observation=False, horizon=400)
        env.reset()
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        action[3] = -1.0
        action[-1] = robot_speech_to_scalar(RobotSpeechToken.ANNOUNCE_HANDOVER)
        info = {}
        for _ in range(20):
            _obs, _reward, _terminated, _truncated, info = env.step(action)
        ready_progress = info["human_reach_progress"]
        self.assertGreaterEqual(info["human_readiness"], info["readiness_threshold"])
        self.assertGreater(ready_progress, 0.0)

        action[-1] = robot_speech_to_scalar(RobotSpeechToken.SILENCE)
        for _ in range(env.readiness_config.readiness_hold_steps + 140):
            _obs, _reward, _terminated, _truncated, info = env.step(action)
        self.assertLess(info["human_readiness"], info["readiness_threshold"])
        self.assertLess(info["human_reach_progress"], ready_progress)

    def test_meaningful_handover_cue_holds_readiness_during_silence(self):
        env = MockAllostaticHandoverEnv(privileged_observation=False, horizon=300)
        env.reset()
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        action[3] = -1.0
        action[-1] = robot_speech_to_scalar(RobotSpeechToken.ANNOUNCE_HANDOVER)
        _obs, _reward, _terminated, _truncated, info = env.step(action)
        self.assertGreater(info["readiness_hold_steps_remaining"], 0)

        action[-1] = robot_speech_to_scalar(RobotSpeechToken.SILENCE)
        for _ in range(env.readiness_config.readiness_hold_steps - 5):
            _obs, _reward, _terminated, _truncated, info = env.step(action)
        self.assertGreaterEqual(info["human_readiness"], info["readiness_threshold"])
        self.assertGreaterEqual(
            info["human_reach_progress"],
            env.readiness_config.min_reach_progress_for_contact,
        )
        self.assertEqual(info["robot_speech_count"], 1)


if __name__ == "__main__":
    unittest.main()
