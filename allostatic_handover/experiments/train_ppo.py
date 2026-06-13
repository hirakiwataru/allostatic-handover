"""Train PPO on mock or human-robot-gym backend."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from allostatic_handover.envs.speech_events import robot_speech_to_scalar
from allostatic_handover.experiments.config_loading import (
    load_mapping,
    merge_config,
    nested_mapping,
    parse_key_value_overrides,
)
from allostatic_handover.experiments.env_factory import make_env
from allostatic_handover.logging.episode_logger import EpisodeLogger
from allostatic_handover.policies import make_scripted_policy
from allostatic_handover.wrappers.hrgym_training_stack import make_sb3_env_from_hrgym
from allostatic_handover.wrappers.robosuite_gymnasium import adapt_robosuite_for_sb3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["mock", "hrgym"], default="mock")
    parser.add_argument("--handover-env", choices=["allostatic", "original"], default="allostatic")
    parser.add_argument("--hrgym-wrapper-stack", choices=["raw", "safe_ik", "safe_ik_air"], default="raw")
    parser.add_argument("--hrgym-shield-type", default=None)
    parser.add_argument("--expert-imitation-alpha", type=float, default=0.25)
    parser.add_argument("--expert-imitation-beta", type=float, default=0.7)
    parser.add_argument("--expert-bc-rollouts", type=int, default=0)
    parser.add_argument("--expert-bc-epochs", type=int, default=0)
    parser.add_argument("--expert-bc-batch-size", type=int, default=256)
    parser.add_argument("--expert-bc-learning-rate", type=float, default=1e-3)
    parser.add_argument("--expert-bc-max-steps", type=int, default=None)
    parser.add_argument("--expert-bc-success-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expert-bc-fallback-to-failed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--expert-bc-motion-loss-weight", type=float, default=1.0)
    parser.add_argument("--expert-bc-gripper-loss-weight", type=float, default=1.0)
    parser.add_argument("--expert-bc-action-std", type=float, default=None)
    parser.add_argument("--expert-dagger-iterations", type=int, default=0)
    parser.add_argument("--expert-dagger-rollouts", type=int, default=5)
    parser.add_argument("--expert-dagger-epochs", type=int, default=0)
    parser.add_argument("--expert-dagger-max-steps", type=int, default=250)
    parser.add_argument("--skip-rl-after-bc", action="store_true")
    parser.add_argument("--expert-bc-speech-policy", default="minimal_speech")
    parser.add_argument("--expert-bc-speech-loss-weight", type=float, default=0.5)
    parser.add_argument("--reward-variant", choices=["task_only", "allostatic", "speech_penalty"], default="task_only")
    parser.add_argument("--reward-config", default=None)
    parser.add_argument("--reward-weight", action="append", default=[])
    parser.add_argument("--allostatic-load-config", default=None)
    parser.add_argument("--allostatic-load-param", action="append", default=[])
    parser.add_argument("--human-fsm-config", default=None)
    parser.add_argument("--human-fsm-param", action="append", default=[])
    parser.add_argument("--total-timesteps", type=int, default=10_000)
    parser.add_argument("--horizon", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--learning-rate", type=float, default=7e-5)
    parser.add_argument("--n-steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--policy-net-arch", default=None)
    parser.add_argument("--activation-fn", choices=["tanh", "relu"], default="tanh")
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--checkpoint-freq", type=int, default=0)
    parser.add_argument("--save-best-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--best-model-min-episodes", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--privileged-observation", action="store_true")
    parser.add_argument("--wandb-mode", choices=["disabled", "offline", "online"], default="disabled")
    parser.add_argument("--wandb-project", default="allostatic-handover-mvp")
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-name", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv_list)
    _apply_config_args(args, argv_list)
    run_name = (
        args.wandb_name
        or f"ppo_{args.backend}_{args.handover_env}_{args.reward_variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = Path(args.output_dir or Path("outputs") / run_name)
    (output_dir / ".mplconfig").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))
    _configure_wandb_dirs(output_dir)

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise ImportError(
            "stable-baselines3 is required for PPO training. Install the training extras from README.md."
        ) from exc

    episode_logger = EpisodeLogger(output_dir)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    raw_env = _make_raw_env(args, seed=args.seed)
    env = Monitor(_make_sb3_env(args, raw_env), filename=str(output_dir / "monitor.csv"))

    class EpisodeMetricsCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.episode_id = 0
            self._episode_values: dict[str, list[float]] = {}
            self._best_success = -1.0
            self._best_return = -float("inf")
            self._training_start_time = 0.0

        def _on_training_start(self) -> None:
            self._training_start_time = time.monotonic()

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            dones = self.locals.get("dones", [])
            for done, info in zip(dones, infos):
                if done:
                    metrics = _episode_metrics_from_done_info(
                        episode_id=self.episode_id,
                        args=args,
                        info=info,
                    )
                    episode_logger.log_episode(metrics)
                    for key, value in metrics.items():
                        if _is_number(value):
                            self._episode_values.setdefault(key, []).append(float(value))
                    self.episode_id += 1
            return True

        def _on_rollout_end(self) -> None:
            self._record_eta_metrics()
            for key, values in self._episode_values.items():
                if values:
                    self.logger.record(f"episode_metrics/{key}", sum(values) / len(values))
            _record_wandb_metric_aliases(self.logger, self._episode_values)
            self._maybe_save_best_model()
            self._episode_values.clear()

        def _record_eta_metrics(self) -> None:
            if self._training_start_time <= 0.0:
                return
            elapsed_seconds = max(0.0, time.monotonic() - self._training_start_time)
            total_timesteps = max(1, int(args.total_timesteps))
            current_timesteps = min(int(self.num_timesteps), total_timesteps)
            progress = min(1.0, current_timesteps / total_timesteps)
            steps_per_second = current_timesteps / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
            remaining_steps = max(0, total_timesteps - current_timesteps)
            eta_seconds = remaining_steps / steps_per_second if steps_per_second > 0.0 else 0.0
            metrics = {
                "elapsed_seconds": elapsed_seconds,
                "eta_seconds": eta_seconds,
                "eta_minutes": eta_seconds / 60.0,
                "eta_hours": eta_seconds / 3600.0,
                "progress_percent": progress * 100.0,
                "steps_per_second": steps_per_second,
                "remaining_timesteps": float(remaining_steps),
            }
            for key, value in metrics.items():
                self.logger.record(f"time/{key}", float(value))
            eta_path = output_dir / "eta.jsonl"
            with eta_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "timesteps": current_timesteps,
                            **metrics,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )

        def _maybe_save_best_model(self) -> None:
            if not args.save_best_model:
                return
            success_values = self._episode_values.get("success", [])
            if len(success_values) < args.best_model_min_episodes:
                return
            mean_success = float(sum(success_values) / len(success_values))
            return_values = self._episode_values.get("return", [])
            mean_return = float(sum(return_values) / len(return_values)) if return_values else 0.0
            if mean_success > self._best_success or (
                mean_success == self._best_success and mean_return > self._best_return
            ):
                self._best_success = mean_success
                self._best_return = mean_return
                best_path = output_dir / "best_model.zip"
                self.model.save(best_path)
                metrics = {
                    "timesteps": int(self.num_timesteps),
                    "mean_success": mean_success,
                    "mean_return": mean_return,
                    "episodes": int(len(success_values)),
                }
                (output_dir / "best_model_metrics.json").write_text(
                    json.dumps(metrics, indent=2),
                    encoding="utf-8",
                )
                print(
                    f"saved best model to {best_path} "
                    f"(success={mean_success:.3f}, return={mean_return:.3f}, timesteps={self.num_timesteps})"
                )

    wandb_run = None
    callbacks: list[Any] = [EpisodeMetricsCallback()]
    if args.checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=args.checkpoint_freq,
                save_path=str(output_dir / "checkpoints"),
                name_prefix="ppo_model",
                save_replay_buffer=False,
                save_vecnormalize=False,
            )
        )
    if args.wandb_mode != "disabled":
        try:
            import wandb
            from wandb.integration.sb3 import WandbCallback
        except ImportError as exc:
            raise ImportError("wandb is required when --wandb-mode is not disabled.") from exc

        wandb_run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group or f"ppo_{args.backend}_{args.handover_env}",
            name=run_name,
            mode=args.wandb_mode,
            config=vars(args),
            sync_tensorboard=True,
            save_code=False,
        )
        callbacks.append(WandbCallback(model_save_path=str(output_dir / "wandb_models"), verbose=2))

    try:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.99,
            gae_lambda=0.9,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            target_kl=args.target_kl,
            policy_kwargs=_policy_kwargs(args),
            verbose=1,
            seed=args.seed,
            tensorboard_log=str(output_dir / "tb"),
            device=args.device,
        )
        if _expert_bc_enabled(args):
            _behavior_clone_from_expert(model=model, args=args, output_dir=output_dir)
        if args.skip_rl_after_bc or args.total_timesteps <= 0:
            print("skipping PPO rollout updates after expert BC")
        else:
            model.learn(total_timesteps=args.total_timesteps, callback=callbacks, log_interval=1)
        model_path = output_dir / "model_final.zip"
        model.save(model_path)
    finally:
        if hasattr(env, "close"):
            env.close()
        if wandb_run is not None:
            wandb_run.finish()
    print(f"saved model to {model_path.resolve()}")
    return 0


def _make_raw_env(args: argparse.Namespace, seed: int | None, render: bool = False):
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
        seed=seed,
        privileged_observation=args.privileged_observation,
        render=render,
        reward_weights=getattr(args, "reward_weights", None),
        allostatic_config=getattr(args, "allostatic_config", None),
        human_fsm_config=getattr(args, "human_fsm_config", None),
        speech_mode=speech_mode,
        **kwargs,
    )


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


def _episode_metrics_from_done_info(
    episode_id: int,
    args: argparse.Namespace,
    info: dict[str, Any],
) -> dict[str, Any]:
    from_info = dict(info.get("episode_metrics") or {})
    episode_info = dict(info.get("episode") or {})
    goal_reached = float(info.get("n_goal_reached", from_info.get("n_goal_reached", 0)) or 0)
    success = float(info.get("success", from_info.get("success", 0.0)) or 0.0)
    if goal_reached > 0:
        success = 1.0
    return {
        "episode_id": episode_id,
        "reward_variant": args.reward_variant,
        "policy": "ppo",
        "success": success,
        "goal_reached": goal_reached,
        "return": float(from_info.get("return", episode_info.get("r", 0.0))),
        "length": int(from_info.get("length", episode_info.get("l", 0))),
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


def _policy_kwargs(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.policy_net_arch is None:
        return None
    layers = [int(part.strip()) for part in args.policy_net_arch.split(",") if part.strip()]
    if not layers:
        return None
    return {
        "net_arch": {"pi": layers, "vf": layers},
        "activation_fn": _activation_fn(args.activation_fn),
    }


def _activation_fn(name: str):
    import torch.nn as nn

    if name == "relu":
        return nn.ReLU
    return nn.Tanh


def _expert_bc_enabled(args: argparse.Namespace) -> bool:
    return args.expert_bc_rollouts > 0 and args.expert_bc_epochs > 0


def _behavior_clone_from_expert(model, args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    _validate_expert_bc_args(args)
    observations, actions, rollout_stats = _collect_expert_bc_dataset(args, output_dir=output_dir)

    print(
        f"expert BC: samples={len(observations)} rollouts={args.expert_bc_rollouts} "
        f"epochs={args.expert_bc_epochs} batch_size={min(args.expert_bc_batch_size, len(observations))}"
    )
    fit_history = [
        _fit_policy_to_expert_actions(
            model=model,
            observations=observations,
            actions=actions,
            args=args,
            epochs=args.expert_bc_epochs,
            label="expert BC",
            seed_offset=7919,
            output_dir=output_dir,
        )
    ]

    dagger_stats: list[dict[str, Any]] = []
    for iteration_id in range(args.expert_dagger_iterations):
        dagger_obs, dagger_actions, stats = _collect_expert_dagger_dataset(
            model=model,
            args=args,
            iteration_id=iteration_id,
            output_dir=output_dir,
        )
        observations = np.concatenate([observations, dagger_obs], axis=0)
        actions = np.concatenate([actions, dagger_actions], axis=0)
        dagger_stats.extend(stats)
        dagger_epochs = args.expert_dagger_epochs or max(1, args.expert_bc_epochs // 2)
        print(
            f"expert DAgger {iteration_id + 1}/{args.expert_dagger_iterations}: "
            f"added={len(dagger_obs)} total={len(observations)} epochs={dagger_epochs}"
        )
        fit_history.append(
            _fit_policy_to_expert_actions(
                model=model,
                observations=observations,
                actions=actions,
                args=args,
                epochs=dagger_epochs,
                label=f"expert DAgger {iteration_id + 1}",
                seed_offset=100_000 + iteration_id,
                output_dir=output_dir,
            )
        )

    if args.expert_bc_action_std is not None:
        _set_policy_action_std(model=model, action_std=args.expert_bc_action_std)

    metrics = {
        "samples": int(len(observations)),
        "rollouts": int(args.expert_bc_rollouts),
        "epochs": int(args.expert_bc_epochs),
        "batch_size": int(max(1, min(args.expert_bc_batch_size, len(observations)))),
        "learning_rate": float(args.expert_bc_learning_rate),
        "motion_loss_weight": float(args.expert_bc_motion_loss_weight),
        "gripper_loss_weight": float(args.expert_bc_gripper_loss_weight),
        "success_only": bool(args.expert_bc_success_only),
        "fallback_to_failed": bool(args.expert_bc_fallback_to_failed),
        "speech_policy": args.expert_bc_speech_policy if args.handover_env == "allostatic" else None,
        "speech_loss_weight": float(args.expert_bc_speech_loss_weight),
        "action_std": None if args.expert_bc_action_std is None else float(args.expert_bc_action_std),
        "initial_loss": fit_history[0]["initial_loss"] if fit_history else None,
        "final_loss": fit_history[-1]["final_loss"] if fit_history else None,
        "fit_history": fit_history,
        "rollout_stats": rollout_stats,
        "dagger_stats": dagger_stats,
    }
    (output_dir / "expert_bc_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _validate_expert_bc_args(args: argparse.Namespace) -> None:
    if args.backend != "hrgym" or args.handover_env not in {"original", "allostatic"}:
        raise ValueError(
            "--expert-bc-rollouts is supported only for --backend hrgym "
            "--handover-env original/allostatic."
        )
    if args.handover_env == "allostatic" and args.hrgym_wrapper_stack not in {"safe_ik", "safe_ik_air"}:
        raise ValueError(
            "--expert-bc-rollouts for allostatic handover requires "
            "--hrgym-wrapper-stack safe_ik or safe_ik_air."
        )
    if args.handover_env == "original" and args.hrgym_wrapper_stack not in {"safe_ik", "safe_ik_air"}:
        raise ValueError("--expert-bc-rollouts requires --hrgym-wrapper-stack safe_ik or safe_ik_air.")


def _collect_expert_bc_dataset(
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    bc_raw_env, bc_env, expert = _make_bc_collection_stack(args, seed=args.seed + 100_000)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    fallback_observations: list[np.ndarray] = []
    fallback_actions: list[np.ndarray] = []
    rollout_stats: list[dict[str, Any]] = []
    max_steps = args.expert_bc_max_steps or args.horizon
    start_time = time.monotonic()

    try:
        for episode_id in range(args.expert_bc_rollouts):
            obs, _ = bc_env.reset()
            speech_policy = _make_bc_speech_policy(args)
            episode_observations: list[np.ndarray] = []
            episode_actions: list[np.ndarray] = []
            total_return = 0.0
            last_info: dict[str, Any] = {}
            step_id = -1
            for step_id in range(max_steps):
                action = np.asarray(expert(_raw_observation_dict(bc_raw_env)), dtype=np.float32)
                action = _append_expert_speech_action(action, args, obs, last_info, speech_policy)
                action = np.clip(action, bc_env.action_space.low, bc_env.action_space.high).astype(np.float32)
                episode_observations.append(np.asarray(obs, dtype=np.float32).copy())
                episode_actions.append(action.copy())
                obs, reward, terminated, truncated, info = bc_env.step(action)
                total_return += float(reward)
                last_info = dict(info)
                if terminated or truncated:
                    break
            success = float(last_info.get("n_goal_reached", 0) > 0 or last_info.get("success", 0.0) > 0)
            fallback_observations.extend(episode_observations)
            fallback_actions.extend(episode_actions)
            if success or not args.expert_bc_success_only:
                observations.extend(episode_observations)
                actions.extend(episode_actions)
            rollout_stats.append(
                {
                    "episode_id": episode_id,
                    "length": int(step_id + 1),
                    "return": float(total_return),
                    "success": success,
                }
            )
            eta = _eta_from_progress(
                start_time=start_time,
                current=episode_id + 1,
                total=args.expert_bc_rollouts,
            )
            print(
                f"expert BC rollout {episode_id + 1}/{args.expert_bc_rollouts}: "
                f"len={step_id + 1} return={total_return:.3f} success={success:.0f} "
                f"elapsed={eta['elapsed_seconds']:.1f}s eta={eta['eta_seconds']:.1f}s "
                f"progress={eta['progress_percent']:.1f}%"
            )
            _record_bc_eta(
                output_dir=output_dir,
                label="expert BC rollout",
                epoch=episode_id + 1,
                epochs=args.expert_bc_rollouts,
                metrics={
                    **eta,
                    "length": float(step_id + 1),
                    "return": float(total_return),
                    "success": float(success),
                },
            )
    finally:
        if hasattr(bc_env, "close"):
            bc_env.close()

    if not observations and args.expert_bc_fallback_to_failed and fallback_observations:
        print("expert BC: no successful samples collected; falling back to all rollout samples")
        observations = fallback_observations
        actions = fallback_actions
    if not observations:
        raise RuntimeError("Expert BC did not collect any successful samples.")
    return np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.float32), rollout_stats


def _collect_expert_dagger_dataset(
    model,
    args: argparse.Namespace,
    iteration_id: int,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    seed = args.seed + 200_000 + iteration_id * 10_000
    bc_raw_env, bc_env, expert = _make_bc_collection_stack(args, seed=seed)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rollout_stats: list[dict[str, Any]] = []
    max_steps = args.expert_dagger_max_steps or args.expert_bc_max_steps or args.horizon
    start_time = time.monotonic()

    try:
        for rollout_id in range(args.expert_dagger_rollouts):
            obs, _ = bc_env.reset()
            speech_policy = _make_bc_speech_policy(args)
            total_return = 0.0
            last_info: dict[str, Any] = {}
            step_id = -1
            for step_id in range(max_steps):
                expert_action = np.asarray(expert(_raw_observation_dict(bc_raw_env)), dtype=np.float32)
                expert_action = _append_expert_speech_action(expert_action, args, obs, last_info, speech_policy)
                expert_action = np.clip(expert_action, bc_env.action_space.low, bc_env.action_space.high).astype(np.float32)
                policy_action, _ = model.predict(obs, deterministic=True)
                policy_action = np.asarray(policy_action, dtype=np.float32).reshape(-1)
                observations.append(np.asarray(obs, dtype=np.float32).copy())
                actions.append(expert_action.copy())
                obs, reward, terminated, truncated, info = bc_env.step(policy_action)
                total_return += float(reward)
                last_info = dict(info)
                if terminated or truncated:
                    break
            success = float(last_info.get("n_goal_reached", 0) > 0 or last_info.get("success", 0.0) > 0)
            rollout_stats.append(
                {
                    "iteration_id": iteration_id,
                    "rollout_id": rollout_id,
                    "length": int(step_id + 1),
                    "return": float(total_return),
                    "success": success,
                }
            )
            eta = _eta_from_progress(
                start_time=start_time,
                current=rollout_id + 1,
                total=args.expert_dagger_rollouts,
            )
            print(
                f"expert DAgger rollout {iteration_id + 1}.{rollout_id + 1}/{args.expert_dagger_rollouts}: "
                f"len={step_id + 1} return={total_return:.3f} success={success:.0f} "
                f"elapsed={eta['elapsed_seconds']:.1f}s eta={eta['eta_seconds']:.1f}s "
                f"progress={eta['progress_percent']:.1f}%"
            )
            _record_bc_eta(
                output_dir=output_dir,
                label=f"expert DAgger rollout {iteration_id + 1}",
                epoch=rollout_id + 1,
                epochs=args.expert_dagger_rollouts,
                metrics={
                    **eta,
                    "length": float(step_id + 1),
                    "return": float(total_return),
                    "success": float(success),
                },
            )
    finally:
        if hasattr(bc_env, "close"):
            bc_env.close()

    if not observations:
        raise RuntimeError("Expert DAgger did not collect any samples.")
    return np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.float32), rollout_stats


def _make_bc_collection_stack(args: argparse.Namespace, seed: int):
    from allostatic_handover.wrappers.hrgym_training_stack import _ScalarPickPlaceHumanCartExpert

    raw_env = _make_raw_env(args, seed=seed)
    bc_args = argparse.Namespace(**vars(args))
    bc_args.hrgym_wrapper_stack = "safe_ik"
    env = _make_sb3_env(bc_args, raw_env)
    expert_action_space = _expert_motor_action_space(env.action_space) if args.handover_env == "allostatic" else env.action_space
    expert = _ScalarPickPlaceHumanCartExpert(
        observation_space=env.observation_space,
        action_space=expert_action_space,
        signal_to_noise_ratio=1.0,
        hover_dist=0.2,
        tan_theta=0.5,
        horizontal_epsilon=0.035,
        vertical_epsilon=0.015,
        goal_dist=0.08,
        gripper_fully_opened_threshold=0.02,
        release_when_delivered=True,
        delta_time=0.01,
        seed=seed,
    )
    return raw_env, env, expert


def _expert_motor_action_space(action_space):
    try:
        from gymnasium import spaces

        return spaces.Box(
            low=np.asarray(action_space.low[:4], dtype=np.float32),
            high=np.asarray(action_space.high[:4], dtype=np.float32),
            dtype=np.float32,
        )
    except ImportError:
        class _Box:
            def __init__(self, source):
                self.low = np.asarray(source.low[:4], dtype=np.float32)
                self.high = np.asarray(source.high[:4], dtype=np.float32)
                self.shape = self.low.shape

        return _Box(action_space)


def _make_bc_speech_policy(args: argparse.Namespace):
    if args.handover_env != "allostatic":
        return None
    return make_scripted_policy(args.expert_bc_speech_policy)


def _append_expert_speech_action(
    motor_action: np.ndarray,
    args: argparse.Namespace,
    obs: np.ndarray,
    info: dict[str, Any],
    speech_policy,
) -> np.ndarray:
    if args.handover_env != "allostatic":
        return motor_action
    if speech_policy is None:
        raise ValueError("Allostatic expert BC requires a speech policy.")
    token = speech_policy.speech(obs, info)
    speech_policy.state.step += 1
    return np.concatenate(
        [
            np.asarray(motor_action, dtype=np.float32).reshape(-1),
            np.array([robot_speech_to_scalar(token)], dtype=np.float32),
        ]
    )


def _fit_policy_to_expert_actions(
    model,
    observations: np.ndarray,
    actions: np.ndarray,
    args: argparse.Namespace,
    epochs: int,
    label: str,
    seed_offset: int,
    output_dir: Path,
) -> dict[str, Any]:
    try:
        import torch as th
    except ImportError as exc:
        raise ImportError("PyTorch is required for --expert-bc-rollouts.") from exc

    obs_tensor = th.as_tensor(observations, dtype=th.float32, device=model.device)
    action_tensor = th.as_tensor(actions, dtype=th.float32, device=model.device)
    weights = th.ones(action_tensor.shape[1], dtype=th.float32, device=model.device)
    weights[:3] *= float(args.expert_bc_motion_loss_weight)
    if action_tensor.shape[1] > 3:
        weights[3] *= float(args.expert_bc_gripper_loss_weight)
    if action_tensor.shape[1] > 4:
        weights[4:] *= float(args.expert_bc_speech_loss_weight)

    policy = model.policy
    policy.set_training_mode(True)
    optimizer = th.optim.Adam(policy.parameters(), lr=args.expert_bc_learning_rate)
    rng = np.random.default_rng(args.seed + seed_offset)
    batch_size = max(1, min(args.expert_bc_batch_size, len(observations)))
    epoch_losses: list[float] = []
    epoch_motion_losses: list[float] = []
    epoch_gripper_losses: list[float] = []
    start_time = time.monotonic()

    for epoch_id in range(epochs):
        losses: list[float] = []
        motion_losses: list[float] = []
        gripper_losses: list[float] = []
        permutation = rng.permutation(len(observations))
        for start in range(0, len(observations), batch_size):
            indices = th.as_tensor(permutation[start : start + batch_size], dtype=th.long, device=model.device)
            distribution = policy.get_distribution(obs_tensor[indices])
            mean_actions = distribution.distribution.mean
            squared_error = (mean_actions - action_tensor[indices]).pow(2)
            loss = (squared_error * weights).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            motion_losses.append(float(squared_error[:, :3].mean().detach().cpu().item()))
            if squared_error.shape[1] > 3:
                gripper_losses.append(float(squared_error[:, 3].mean().detach().cpu().item()))
        mean_loss = float(np.mean(losses))
        mean_motion_loss = float(np.mean(motion_losses))
        mean_gripper_loss = float(np.mean(gripper_losses)) if gripper_losses else 0.0
        epoch_losses.append(mean_loss)
        epoch_motion_losses.append(mean_motion_loss)
        epoch_gripper_losses.append(mean_gripper_loss)
        if epoch_id == 0 or (epoch_id + 1) % 25 == 0 or epoch_id == epochs - 1:
            eta = _eta_from_progress(start_time=start_time, current=epoch_id + 1, total=epochs)
            print(
                f"{label} epoch {epoch_id + 1}/{epochs}: "
                f"loss={mean_loss:.6f} motion={mean_motion_loss:.6f} gripper={mean_gripper_loss:.6f} "
                f"elapsed={eta['elapsed_seconds']:.1f}s eta={eta['eta_seconds']:.1f}s "
                f"progress={eta['progress_percent']:.1f}%"
            )
            _record_bc_eta(
                output_dir=output_dir,
                label=label,
                epoch=epoch_id + 1,
                epochs=epochs,
                metrics={
                    **eta,
                    "loss": mean_loss,
                    "motion_loss": mean_motion_loss,
                    "gripper_loss": mean_gripper_loss,
                },
            )

    return {
        "label": label,
        "samples": int(len(observations)),
        "epochs": int(epochs),
        "initial_loss": epoch_losses[0] if epoch_losses else None,
        "final_loss": epoch_losses[-1] if epoch_losses else None,
        "final_motion_loss": epoch_motion_losses[-1] if epoch_motion_losses else None,
        "final_gripper_loss": epoch_gripper_losses[-1] if epoch_gripper_losses else None,
    }


def _eta_from_progress(start_time: float, current: int, total: int) -> dict[str, float]:
    elapsed_seconds = max(0.0, time.monotonic() - start_time)
    total = max(1, int(total))
    current = min(max(0, int(current)), total)
    progress = current / total
    items_per_second = current / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
    remaining = max(0, total - current)
    eta_seconds = remaining / items_per_second if items_per_second > 0.0 else 0.0
    return {
        "elapsed_seconds": elapsed_seconds,
        "eta_seconds": eta_seconds,
        "eta_minutes": eta_seconds / 60.0,
        "eta_hours": eta_seconds / 3600.0,
        "progress_percent": progress * 100.0,
        "items_per_second": items_per_second,
        "remaining_items": float(remaining),
    }


def _record_bc_eta(
    output_dir: Path,
    label: str,
    epoch: int,
    epochs: int,
    metrics: dict[str, float],
) -> None:
    record = {
        "label": label,
        "epoch": int(epoch),
        "epochs": int(epochs),
        **metrics,
    }
    with (output_dir / "expert_bc_eta.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")
    try:
        import wandb

        if wandb.run is not None:
            prefix = "bc" if label == "expert BC" else f"bc/{label.replace(' ', '_').lower()}"
            wandb.log({f"{prefix}/{key}": value for key, value in record.items() if _is_number(value)})
    except ImportError:
        pass


def _set_policy_action_std(model, action_std: float) -> None:
    if action_std <= 0:
        raise ValueError("--expert-bc-action-std must be positive.")
    policy = model.policy
    if not hasattr(policy, "log_std"):
        raise AttributeError("The current PPO policy does not expose log_std.")
    import torch as th

    with th.no_grad():
        policy.log_std.data.fill_(float(np.log(action_std)))
    print(f"set PPO policy action std to {action_std}")


def _raw_observation_dict(raw_env) -> dict[str, Any]:
    if hasattr(raw_env, "_get_observations"):
        return raw_env._get_observations(force_update=True)
    raise AttributeError("human-robot-gym environment does not expose _get_observations(force_update=True).")


def _configure_wandb_dirs(output_dir: Path) -> None:
    wandb_root = output_dir / "wandb"
    wandb_root.mkdir(parents=True, exist_ok=True)
    (wandb_root / "cache").mkdir(parents=True, exist_ok=True)
    (wandb_root / "config").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_DIR", str(wandb_root))
    os.environ.setdefault("WANDB_CACHE_DIR", str(wandb_root / "cache"))
    os.environ.setdefault("WANDB_CONFIG_DIR", str(wandb_root / "config"))


def _record_wandb_metric_aliases(logger, episode_values: dict[str, list[float]]) -> None:
    def mean(key: str, default: float = 0.0) -> float:
        values = episode_values.get(key, [])
        if not values:
            return default
        return float(sum(values) / len(values))

    aliases = {
        "train/episode_reward": mean("return"),
        "train/episode_length": mean("length"),
        "train/success_rate": mean("success"),
        "handover/time_to_success": mean("handover_time"),
        "handover/drop_rate": mean("drop_count"),
        "handover/collision_count": mean("collision_count"),
        "handover/object_in_human_hand": mean("object_in_human_hand"),
        "speech/robot_speech_count": mean("robot_speech_count"),
        "speech/silence_ratio": mean("silence_ratio"),
        "speech/repeated_speech_count": mean("repeated_speech_count"),
        "human_state/ready_ratio": mean("human_state_ready_ratio"),
        "human_state/hesitant_ratio": mean("human_state_hesitant_ratio"),
        "human_state/distracted_ratio": mean("human_state_distracted_ratio"),
        "human_state/overloaded_ratio": mean("human_state_overloaded_ratio"),
        "human_state/withdrawing_ratio": mean("human_state_withdrawing_ratio"),
        "human_state/grasping_ratio": mean("human_state_grasping_ratio"),
        "allostasis/load_mean": mean("allostatic_load_mean"),
        "allostasis/load_max": mean("allostatic_load_max"),
        "allostasis/load_final": mean("allostatic_load_final"),
        "allostasis/load_area": mean("allostatic_load_area"),
        "allostasis/attention_load": mean("attention_load"),
        "allostasis/turn_taking_load": mean("turn_taking_load"),
        "allostasis/proxemic_stress": mean("proxemic_stress"),
        "allostasis/motor_adaptation_cost": mean("motor_adaptation_cost"),
        "allostasis/human_waiting_cost": mean("human_waiting_time"),
        "allostasis/human_reach_effort": mean("human_reach_effort"),
        "human/readiness_mean": mean("human_readiness_mean"),
        "human/readiness_final": mean("human_readiness_final"),
        "human/readiness_blocked_count": mean("readiness_blocked_count"),
        "human/readiness_hold_steps_remaining": mean("readiness_hold_steps_remaining"),
        "human/readiness_belief": mean("human_readiness_belief"),
        "human/reach_progress": mean("human_reach_progress"),
        "human/reach_progress_mean": mean("human_reach_progress_mean"),
        "human/animation_gated_by_readiness": mean("animation_gated_by_readiness"),
        "human/reach_out_started_count": mean("reach_out_started_count"),
        "allostasis/load_proxy": mean("allostatic_load_proxy"),
    }
    for key, value in aliases.items():
        logger.record(key, value)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


if __name__ == "__main__":
    raise SystemExit(main())
