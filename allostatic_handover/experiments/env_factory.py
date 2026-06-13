"""Environment factory for CLI experiments."""

from __future__ import annotations

import json
from typing import Any

from allostatic_handover.envs.mock_handover_env import MockAllostaticHandoverEnv


def make_env(
    backend: str = "mock",
    reward_variant: str = "task_only",
    handover_env: str = "allostatic",
    horizon: int = 160,
    seed: int | None = None,
    privileged_observation: bool = False,
    render: bool = False,
    reward_weights: dict[str, Any] | None = None,
    allostatic_config: dict[str, Any] | None = None,
    human_fsm_config: dict[str, Any] | None = None,
    speech_mode: str | None = None,
    **kwargs: Any,
):
    backend = backend.strip().lower()
    if backend == "mock":
        return MockAllostaticHandoverEnv(
            reward_variant=reward_variant,
            horizon=horizon,
            seed=seed,
            privileged_observation=privileged_observation,
            reward_weights=reward_weights,
            **kwargs,
        )
    if backend == "hrgym":
        env_kwargs = {
            "horizon": horizon,
            "seed": 0 if seed is None else seed,
            "use_camera_obs": False,
            "use_object_obs": True,
            "has_renderer": bool(render),
            "has_offscreen_renderer": False,
            "render_camera": None,
            "renderer": "mjviewer",
            "render_collision_mesh": False,
            "hard_reset": False,
            "done_at_collision": False,
            "done_at_success": True,
            "shield_type": "OFF",
            "robots": "Schunk",
        }
        env_kwargs["controller_configs"] = _hrgym_controller_configs("schunk")
        env_kwargs.update(kwargs)

        handover_env = handover_env.strip().lower()
        if handover_env == "allostatic":
            from allostatic_handover.registration import make_hrgym_env

            env_kwargs["reward_variant"] = reward_variant
            env_kwargs["privileged_observation"] = privileged_observation
            env_kwargs["speech_mode"] = speech_mode or env_kwargs.get("speech_mode", "learned")
            if reward_weights is not None:
                env_kwargs["reward_weights"] = reward_weights
            if allostatic_config is not None:
                env_kwargs["allostatic_config"] = allostatic_config
            if human_fsm_config is not None:
                env_kwargs["human_fsm_config"] = human_fsm_config
            return make_hrgym_env(**env_kwargs)
        if handover_env == "original":
            from allostatic_handover.registration import make_original_handover_env

            return make_original_handover_env(**env_kwargs)
        raise ValueError("handover_env must be 'allostatic' or 'original'")
    raise ValueError("backend must be 'mock' or 'hrgym'")


def _hrgym_controller_configs(robot_name: str):
    from robosuite.controllers import load_composite_controller_config
    from human_robot_gym.utils.mjcf_utils import file_path_completion, merge_configs

    controller_config_path = file_path_completion("controllers/failsafe_controller/config/failsafe.json")
    robot_config_path = file_path_completion(f"models/robots/config/{robot_name.lower()}.json")
    failsafe_config = load_composite_controller_config(controller=controller_config_path)
    with open(robot_config_path, encoding="utf-8") as f:
        robot_config = json.load(f)
    return [
        {
            "body_parts": {
                "right": merge_configs(failsafe_config["body_parts"]["right"], robot_config),
            }
        }
    ]


def reset_env(env):
    result = env.reset()
    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, {}


def step_env(env, action):
    result = env.step(action)
    if isinstance(result, tuple) and len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, reward, bool(terminated or truncated), info
    obs, reward, done, info = result
    return obs, reward, bool(done), info
