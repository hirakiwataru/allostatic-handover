"""Evaluate a saved PPO policy on mock or human-robot-gym backend."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from allostatic_handover.experiments.config_loading import (
    load_mapping,
    merge_config,
    nested_mapping,
    parse_key_value_overrides,
)
from allostatic_handover.experiments.env_factory import make_env
from allostatic_handover.experiments.run_scripted_rollouts import _format_step_info, _should_print_step
from allostatic_handover.logging.episode_logger import EpisodeLogger
from allostatic_handover.wrappers.hrgym_training_stack import make_sb3_env_from_hrgym
from allostatic_handover.wrappers.robosuite_gymnasium import adapt_robosuite_for_sb3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--backend", choices=["mock", "hrgym"], default="hrgym")
    parser.add_argument("--handover-env", choices=["allostatic", "original"], default="allostatic")
    parser.add_argument("--hrgym-wrapper-stack", choices=["raw", "safe_ik", "safe_ik_air"], default="raw")
    parser.add_argument("--hrgym-shield-type", default=None)
    parser.add_argument("--expert-imitation-alpha", type=float, default=0.25)
    parser.add_argument("--expert-imitation-beta", type=float, default=0.7)
    parser.add_argument("--reward-variant", choices=["task_only", "allostatic", "speech_penalty"], default="task_only")
    parser.add_argument("--reward-config", default=None)
    parser.add_argument("--reward-weight", action="append", default=[])
    parser.add_argument("--allostatic-load-config", default=None)
    parser.add_argument("--allostatic-load-param", action="append", default=[])
    parser.add_argument("--human-fsm-config", default=None)
    parser.add_argument("--human-fsm-param", action="append", default=[])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=160)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--print-step-info", action="store_true")
    parser.add_argument("--print-interval", type=int, default=10)
    parser.add_argument("--privileged-observation", action="store_true")
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv_list)
    _apply_config_args(args, argv_list)
    run_name = f"eval_ppo_{args.backend}_{args.reward_variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir or Path("outputs") / run_name)
    (output_dir / ".mplconfig").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))
    logger = EpisodeLogger(output_dir)

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise ImportError(
            "stable-baselines3 is required for PPO evaluation. Install the training extras from README.md."
        ) from exc

    raw_env = _make_raw_env(args)
    env = _make_sb3_env(args, raw_env)
    model = PPO.load(args.model_path, env=env, device=args.device)

    try:
        for episode_id in range(args.episodes):
            obs, _reset_info = env.reset()
            total_return = 0.0
            last_info: dict[str, Any] = {}
            step_id = -1
            for step_id in range(args.horizon):
                action, _state = model.predict(obs, deterministic=args.deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                total_return += float(reward)
                last_info = dict(info)
                logger.log_step(episode_id, step_id, obs, action, reward, done, last_info)
                if args.render and hasattr(env, "render"):
                    env.render()
                if args.print_step_info and _should_print_step(step_id, last_info, args.print_interval):
                    print(_format_step_info(episode_id, step_id, reward, last_info))
                if done:
                    break

            metrics = _episode_metrics(
                episode_id=episode_id,
                args=args,
                total_return=total_return,
                steps=step_id + 1,
                info=last_info,
            )
            logger.log_episode(metrics)
            print(
                f"episode={episode_id} success={metrics['success']} "
                f"return={metrics['return']:.3f} load={metrics['allostatic_load_total']:.3f} "
                f"speech={metrics['robot_speech_count']}"
            )
    finally:
        if hasattr(env, "close"):
            env.close()

    print(f"wrote evaluation logs to {output_dir.resolve()}")
    return 0


def _make_raw_env(args: argparse.Namespace):
    kwargs: dict[str, Any] = {}
    if args.backend == "hrgym" and args.hrgym_shield_type:
        kwargs["shield_type"] = args.hrgym_shield_type
    speech_mode = None
    if args.backend == "hrgym" and args.handover_env == "allostatic" and args.hrgym_wrapper_stack == "safe_ik":
        speech_mode = "external"
    return make_env(
        backend=args.backend,
        handover_env=args.handover_env,
        reward_variant=args.reward_variant,
        horizon=args.horizon,
        seed=args.seed,
        privileged_observation=args.privileged_observation,
        render=args.render,
        reward_weights=getattr(args, "reward_weights", None),
        allostatic_config=getattr(args, "allostatic_config", None),
        human_fsm_config=getattr(args, "human_fsm_config", None),
        speech_mode=speech_mode,
        **kwargs,
    )


def _apply_config_args(args: argparse.Namespace, argv: list[str]) -> None:
    reward_config = load_mapping(args.reward_config)
    if reward_config and not _arg_was_provided(argv, "--reward-variant") and reward_config.get("reward_variant"):
        args.reward_variant = str(reward_config["reward_variant"])
    args.reward_weights = merge_config(
        nested_mapping(reward_config, "weights"),
        parse_key_value_overrides(args.reward_weight),
    )

    allostatic_config = load_mapping(args.allostatic_load_config)
    if "allostatic_load" in allostatic_config:
        allostatic_config = nested_mapping(allostatic_config, "allostatic_load")
    args.allostatic_config = merge_config(
        allostatic_config,
        parse_key_value_overrides(args.allostatic_load_param),
    )

    human_fsm_config = load_mapping(args.human_fsm_config)
    if "fsm" in human_fsm_config:
        human_fsm_config = {
            **nested_mapping(human_fsm_config, "fsm"),
            **({"readiness": nested_mapping(human_fsm_config, "readiness")} if "readiness" in human_fsm_config else {}),
        }
    args.human_fsm_config = merge_config(
        human_fsm_config,
        parse_key_value_overrides(args.human_fsm_param),
    )


def _arg_was_provided(argv: list[str], flag: str) -> bool:
    return any(item == flag or item.startswith(f"{flag}=") for item in argv)


def _episode_metrics(
    episode_id: int,
    args: argparse.Namespace,
    total_return: float,
    steps: int,
    info: dict[str, Any],
) -> dict[str, Any]:
    from_info = dict(info.get("episode_metrics") or {})
    goal_reached = float(info.get("n_goal_reached", from_info.get("n_goal_reached", 0)) or 0)
    success = float(info.get("success", from_info.get("success", 0.0)) or 0.0)
    if goal_reached > 0:
        success = 1.0
    return {
        "episode_id": episode_id,
        "reward_variant": args.reward_variant,
        "policy": "ppo_eval",
        "success": success,
        "goal_reached": goal_reached,
        "return": float(from_info.get("return", total_return)),
        "length": int(from_info.get("length", steps)),
        "handover_time": from_info.get("handover_time", info.get("handover_time")),
        "robot_speech_count": int(from_info.get("robot_speech_count", info.get("robot_speech_count", 0))),
        "silence_count": int(from_info.get("silence_count", info.get("silence_count", 0))),
        "silence_ratio": float(from_info.get("silence_ratio", info.get("silence_ratio", 0.0))),
        "repeated_speech_count": int(
            from_info.get("repeated_speech_count", info.get("repeated_speech_count", 0))
        ),
        "human_waiting_time": float(from_info.get("human_waiting_time", info.get("human_waiting_time", 0.0))),
        "human_reach_effort": float(from_info.get("human_reach_effort", info.get("human_reach_effort_sum", 0.0))),
        "human_readiness": float(from_info.get("human_readiness", info.get("human_readiness", 0.0))),
        "human_readiness_belief": float(
            from_info.get("human_readiness_belief", info.get("human_readiness_belief", 0.0))
        ),
        "human_readiness_mean": float(
            from_info.get("human_readiness_mean", info.get("human_readiness_mean", 0.0))
        ),
        "human_readiness_min": float(from_info.get("human_readiness_min", info.get("human_readiness_min", 0.0))),
        "human_readiness_max": float(from_info.get("human_readiness_max", info.get("human_readiness_max", 0.0))),
        "human_readiness_final": float(
            from_info.get("human_readiness_final", info.get("human_readiness_final", info.get("human_readiness", 0.0)))
        ),
        "readiness_threshold": float(from_info.get("readiness_threshold", info.get("readiness_threshold", 0.0))),
        "readiness_blocked_count": int(
            from_info.get("readiness_blocked_count", info.get("readiness_blocked_count", 0))
        ),
        "readiness_hold_steps_remaining": int(
            from_info.get("readiness_hold_steps_remaining", info.get("readiness_hold_steps_remaining", 0))
        ),
        "human_reach_progress": float(from_info.get("human_reach_progress", info.get("human_reach_progress", 0.0))),
        "human_reach_progress_mean": float(
            from_info.get("human_reach_progress_mean", info.get("human_reach_progress_mean", 0.0))
        ),
        "animation_gated_by_readiness": float(
            from_info.get("animation_gated_by_readiness", info.get("animation_gated_by_readiness", 0.0))
        ),
        "reach_out_started_count": int(
            from_info.get("reach_out_started_count", info.get("reach_out_started_count", 0))
        ),
        "allostatic_load_proxy": float(from_info.get("allostatic_load_proxy", info.get("allostatic_load_proxy", 0.0))),
        "allostatic_load_total": float(
            from_info.get("allostatic_load_total", info.get("allostatic_load_total", 0.0))
        ),
        "allostatic_load_mean": float(
            from_info.get("allostatic_load_mean", info.get("allostatic_load_mean", 0.0))
        ),
        "allostatic_load_max": float(from_info.get("allostatic_load_max", info.get("allostatic_load_max", 0.0))),
        "allostatic_load_final": float(
            from_info.get(
                "allostatic_load_final",
                info.get("allostatic_load_final", info.get("allostatic_load_total", 0.0)),
            )
        ),
        "allostatic_load_area": float(
            from_info.get("allostatic_load_area", info.get("allostatic_load_area", 0.0))
        ),
        "attention_load": float(from_info.get("attention_load", info.get("attention_load", 0.0))),
        "turn_taking_load": float(from_info.get("turn_taking_load", info.get("turn_taking_load", 0.0))),
        "proxemic_stress": float(from_info.get("proxemic_stress", info.get("proxemic_stress", 0.0))),
        "motor_adaptation_cost": float(
            from_info.get("motor_adaptation_cost", info.get("motor_adaptation_cost", 0.0))
        ),
        "annoyance": float(from_info.get("annoyance", info.get("annoyance", 0.0))),
        "trust": float(from_info.get("trust", info.get("trust", 0.0))),
        "withdrawal_count": int(from_info.get("withdrawal_count", info.get("withdrawal_count", 0))),
        "overload_count": int(from_info.get("overload_count", info.get("overload_count", 0))),
        "collision_count": int(from_info.get("collision_count", info.get("collision_count", info.get("n_collisions", 0)))),
        "drop_count": int(from_info.get("drop_count", info.get("drop_count", 0))),
        "object_in_human_hand": float(from_info.get("object_in_human_hand", info.get("object_in_human_hand", 0.0))),
        "human_state_ready_ratio": float(
            from_info.get("human_state_ready_ratio", info.get("human_state_ready_ratio", 0.0))
        ),
        "human_state_hesitant_ratio": float(
            from_info.get("human_state_hesitant_ratio", info.get("human_state_hesitant_ratio", 0.0))
        ),
        "human_state_distracted_ratio": float(
            from_info.get("human_state_distracted_ratio", info.get("human_state_distracted_ratio", 0.0))
        ),
        "human_state_overloaded_ratio": float(
            from_info.get("human_state_overloaded_ratio", info.get("human_state_overloaded_ratio", 0.0))
        ),
        "human_state_withdrawing_ratio": float(
            from_info.get("human_state_withdrawing_ratio", info.get("human_state_withdrawing_ratio", 0.0))
        ),
        "human_state_grasping_ratio": float(
            from_info.get("human_state_grasping_ratio", info.get("human_state_grasping_ratio", 0.0))
        ),
    }


def _make_sb3_env(args: argparse.Namespace, raw_env):
    if args.backend != "hrgym":
        return adapt_robosuite_for_sb3(raw_env)
    return make_sb3_env_from_hrgym(
        raw_env,
        handover_env=args.handover_env,
        wrapper_stack=args.hrgym_wrapper_stack,
        expert_imitation_alpha=args.expert_imitation_alpha,
        expert_imitation_beta=args.expert_imitation_beta,
    )


if __name__ == "__main__":
    raise SystemExit(main())
