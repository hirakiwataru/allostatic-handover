"""Compare a PPO policy against the RobotHumanHandoverCart expert."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from allostatic_handover.experiments.env_factory import make_env
from allostatic_handover.wrappers.hrgym_training_stack import _ScalarPickPlaceHumanCartExpert, make_sb3_env_from_hrgym


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hrgym-shield-type", default="PFL")
    parser.add_argument("--rollout-policy", choices=["ppo", "expert"], default="ppo")
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise ImportError("stable-baselines3 is required for policy diagnostics.") from exc

    raw_env = make_env(
        backend="hrgym",
        handover_env="original",
        reward_variant="task_only",
        horizon=1000,
        seed=args.seed,
        privileged_observation=False,
        render=False,
        shield_type=args.hrgym_shield_type,
    )
    env = make_sb3_env_from_hrgym(raw_env, handover_env="original", wrapper_stack="safe_ik")
    model = PPO.load(Path(args.model_path), env=env, device=args.device)
    expert = _ScalarPickPlaceHumanCartExpert(
        observation_space=env.observation_space,
        action_space=env.action_space,
        signal_to_noise_ratio=1.0,
        hover_dist=0.2,
        tan_theta=0.5,
        horizontal_epsilon=0.035,
        vertical_epsilon=0.015,
        goal_dist=0.08,
        gripper_fully_opened_threshold=0.02,
        release_when_delivered=True,
        delta_time=0.01,
        seed=args.seed,
    )

    try:
        obs, _ = env.reset()
        total_return = 0.0
        for step_id in range(args.steps):
            raw_obs = raw_env._get_observations(force_update=True)
            expert_action = np.asarray(expert(raw_obs), dtype=np.float32)
            ppo_action, _ = model.predict(obs, deterministic=True)
            ppo_action = np.asarray(ppo_action, dtype=np.float32).reshape(-1)
            action = expert_action if args.rollout_policy == "expert" else ppo_action
            obs, reward, terminated, truncated, info = env.step(action)
            total_return += float(reward)
            print(
                f"step={step_id:03d} "
                f"dist_obj={_scalar(raw_obs.get('dist_eef_to_object')):.4f} "
                f"dist_target={_scalar(raw_obs.get('dist_object_to_target')):.4f} "
                f"gripped={_scalar(raw_obs.get('object_gripped')):.0f} "
                f"expert={_format_action(expert_action)} "
                f"ppo={_format_action(ppo_action)} "
                f"diff_motion={np.linalg.norm(expert_action[:3] - ppo_action[:3]):.4f} "
                f"diff_grip={abs(float(expert_action[3] - ppo_action[3])):.4f} "
                f"reward={float(reward):.3f}"
            )
            if terminated or truncated:
                print(
                    f"done step={step_id} return={total_return:.3f} "
                    f"n_goal_reached={info.get('n_goal_reached', 0)} success={info.get('success', 0)}"
                )
                break
    finally:
        env.close()
    return 0


def _scalar(value) -> float:
    if value is None:
        return float("nan")
    return float(np.asarray(value).reshape(-1)[0])


def _format_action(action: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(v):+.3f}" for v in action.reshape(-1)) + "]"


if __name__ == "__main__":
    raise SystemExit(main())
