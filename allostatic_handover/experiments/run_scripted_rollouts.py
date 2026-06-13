"""Run scripted rollout comparisons and log CSV/JSONL."""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from allostatic_handover.experiments.env_factory import make_env, reset_env, step_env
from allostatic_handover.logging.episode_logger import EpisodeLogger
from allostatic_handover.logging.wandb_logger import WandbRun
from allostatic_handover.policies import make_scripted_policy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["mock", "hrgym"], default="mock")
    parser.add_argument("--policy", default="minimal_speech")
    parser.add_argument("--reward-variant", choices=["task_only", "allostatic", "speech_penalty"], default="task_only")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--privileged-observation", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-sleep", type=float, default=0.02)
    parser.add_argument("--print-step-info", action="store_true")
    parser.add_argument("--print-interval", type=int, default=10)
    parser.add_argument("--wandb-mode", choices=["disabled", "offline", "online"], default="disabled")
    parser.add_argument("--wandb-project", default="allostatic-handover-mvp")
    parser.add_argument("--wandb-group", default="scripted_degeneracy_check")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_name = _run_name(args)
    output_dir = Path(args.output_dir or Path("outputs") / run_name)
    logger = EpisodeLogger(output_dir)
    wandb = WandbRun(
        enabled=args.wandb_mode != "disabled",
        project=args.wandb_project,
        group=args.wandb_group,
        mode=args.wandb_mode,
        name=run_name,
        config=vars(args),
        tags=["mvp", "scripted", args.reward_variant, args.policy],
    )

    env = make_env(
        backend=args.backend,
        reward_variant=args.reward_variant,
        horizon=args.horizon,
        seed=args.seed,
        privileged_observation=args.privileged_observation,
        render=args.render,
    )
    policy = make_scripted_policy(args.policy)

    try:
        for episode_id in range(args.episodes):
            obs, info = reset_env(env)
            if args.render:
                _render_env(env, args.render_sleep)
            policy.reset()
            total_return = 0.0
            last_info: dict[str, Any] = dict(info)
            for step_id in range(args.horizon):
                action = policy.act(obs, env, last_info)
                obs, reward, done, info = step_env(env, action)
                if args.render:
                    _render_env(env, args.render_sleep)
                total_return += float(reward)
                last_info = dict(info)
                logger.log_step(episode_id, step_id, obs, action, reward, done, last_info)
                wandb.log(_wandb_step_metrics(last_info, reward), step=episode_id * args.horizon + step_id)
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
            wandb.log({f"episode/{k}": v for k, v in metrics.items() if _is_number(v)})
            print(
                f"episode={episode_id} success={metrics['success']} "
                f"return={metrics['return']:.3f} load={metrics['allostatic_load_total']:.3f} "
                f"speech={metrics['robot_speech_count']}"
            )
    finally:
        if hasattr(env, "close"):
            env.close()
        wandb.finish()

    print(f"wrote logs to {output_dir.resolve()}")
    return 0


def _run_name(args: argparse.Namespace) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{args.backend}_{args.policy}_{args.reward_variant}_{stamp}"


def _episode_metrics(
    episode_id: int,
    args: argparse.Namespace,
    total_return: float,
    steps: int,
    info: dict[str, Any],
) -> dict[str, Any]:
    from_info = dict(info.get("episode_metrics") or {})
    metrics = {
        "episode_id": episode_id,
        "reward_variant": args.reward_variant,
        "policy": args.policy,
        "success": float(info.get("success", from_info.get("success", 0.0))),
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
    return metrics


def _wandb_step_metrics(info: dict[str, Any], reward: float) -> dict[str, float]:
    keys = [
        "allostatic_load_total",
        "attention_load",
        "turn_taking_load",
        "proxemic_stress",
        "motor_adaptation_cost",
        "annoyance",
        "trust",
        "robot_speech_count",
        "silence_ratio",
        "repeated_speech_count",
        "human_waiting_time",
        "human_reach_effort",
        "allostatic_load_mean",
        "allostatic_load_max",
        "allostatic_load_area",
        "allostatic_load_proxy",
        "human_readiness",
        "human_readiness_belief",
        "human_readiness_mean",
        "human_readiness_final",
        "readiness_blocked_count",
        "readiness_hold_steps_remaining",
        "human_reach_progress",
        "human_reach_progress_mean",
        "animation_gated_by_readiness",
        "reach_out_started_count",
        "human_state_ready_ratio",
        "human_state_hesitant_ratio",
        "human_state_overloaded_ratio",
        "human_state_withdrawing_ratio",
        "withdrawal_count",
        "overload_count",
    ]
    metrics = {"step/reward": float(reward)}
    for key in keys:
        value = info.get(key)
        if _is_number(value):
            metrics[f"step/{key}"] = float(value)
    return metrics


def _render_env(env, sleep_seconds: float) -> None:
    if hasattr(env, "render"):
        env.render()
    if sleep_seconds > 0.0:
        time.sleep(sleep_seconds)


def _should_print_step(step_id: int, info: dict[str, Any], interval: int) -> bool:
    if interval > 0 and step_id % interval == 0:
        return True
    return bool(info.get("conversation"))


def _format_step_info(episode_id: int, step_id: int, reward: float, info: dict[str, Any]) -> str:
    messages = []
    for event in info.get("conversation") or []:
        speaker = event.get("speaker", "?")
        text = event.get("text", "")
        if text:
            messages.append(f"{speaker}: {text}")

    message = " | ".join(messages) if messages else "-"
    load = float(info.get("allostatic_load_total", 0.0))
    readiness = float(info.get("human_readiness", 0.0))
    reach_progress = float(info.get("human_reach_progress", 0.0))
    hold_steps = int(info.get("readiness_hold_steps_remaining", 0))
    state = info.get("human_state", "?")
    robot = info.get("robot_speech", "silence")
    human = info.get("human_speech_event", "silence")
    return (
        f"episode={episode_id} step={step_id} reward={float(reward):.3f} "
        f"state={state} readiness={readiness:.3f} reach={reach_progress:.3f} hold={hold_steps} "
        f"load={load:.3f} robot={robot} human={human} "
        f"conversation={message}"
    )


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


if __name__ == "__main__":
    raise SystemExit(main())
