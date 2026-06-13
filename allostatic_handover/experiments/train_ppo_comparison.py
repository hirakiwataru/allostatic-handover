"""Train and evaluate task-only, speech-penalty, and allostatic PPO runs."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from allostatic_handover.experiments.eval_ppo import main as eval_ppo_main
from allostatic_handover.experiments.train_ppo import main as train_ppo_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["mock", "hrgym"], default="hrgym")
    parser.add_argument("--handover-env", choices=["allostatic", "original"], default="allostatic")
    parser.add_argument("--hrgym-wrapper-stack", choices=["raw", "safe_ik", "safe_ik_air"], default="safe_ik_air")
    parser.add_argument("--hrgym-shield-type", default="PFL")
    parser.add_argument("--reward-variants", default="task_only,speech_penalty,allostatic")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--best-model-min-episodes", type=int, default=3)
    parser.add_argument("--wandb-mode", choices=["disabled", "offline", "online"], default="online")
    parser.add_argument("--wandb-project", default="allostatic-handover-mvp")
    parser.add_argument("--wandb-group", default="ppo_allostatic_readiness_hold_eta_air_compare")
    parser.add_argument("--reward-config", default=None)
    parser.add_argument("--allostatic-load-config", default=None)
    parser.add_argument("--human-fsm-config", default=None)
    parser.add_argument("--expert-bc-rollouts", type=int, default=30)
    parser.add_argument("--expert-bc-epochs", type=int, default=300)
    parser.add_argument("--expert-bc-batch-size", type=int, default=512)
    parser.add_argument("--expert-bc-learning-rate", type=float, default=0.001)
    parser.add_argument("--expert-bc-success-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expert-bc-fallback-to-failed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expert-bc-motion-loss-weight", type=float, default=2.0)
    parser.add_argument("--expert-bc-gripper-loss-weight", type=float, default=2.0)
    parser.add_argument("--expert-bc-speech-policy", default="excessive_speech")
    parser.add_argument("--expert-bc-speech-loss-weight", type=float, default=0.5)
    parser.add_argument("--expert-bc-action-std", type=float, default=0.05)
    parser.add_argument("--expert-dagger-iterations", type=int, default=0)
    parser.add_argument("--expert-dagger-rollouts", type=int, default=3)
    parser.add_argument("--expert-dagger-epochs", type=int, default=200)
    parser.add_argument("--expert-dagger-max-steps", type=int, default=250)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    variants = [item.strip() for item in args.reward_variants.split(",") if item.strip()]
    if not variants:
        raise ValueError("--reward-variants must include at least one variant.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root or Path("outputs") / f"ppo_compare_safe_ik_{stamp}")
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    for variant in variants:
        run_name = f"{args.handover_env}_{variant}_seed{args.seed}"
        train_dir = output_root / run_name
        eval_dir = output_root / f"eval_{run_name}"
        train_argv = _train_argv(args, variant, train_dir, run_name)
        print(f"=== training {variant}: {train_dir} ===")
        train_ppo_main(train_argv)

        model_path = train_dir / "best_model.zip"
        if not model_path.exists():
            model_path = train_dir / "model_final.zip"
        print(f"=== evaluating {variant}: {model_path} ===")
        eval_ppo_main(_eval_argv(args, variant, model_path, eval_dir))
        summary_rows.append(_summarize_eval_csv(variant, eval_dir / "episodes.csv"))

    _write_summary(output_root / "comparison_summary.csv", summary_rows)
    print(f"comparison summary written to {(output_root / 'comparison_summary.csv').resolve()}")
    return 0


def _train_argv(args: argparse.Namespace, variant: str, output_dir: Path, run_name: str) -> list[str]:
    values = [
        "--backend",
        args.backend,
        "--handover-env",
        args.handover_env,
        "--hrgym-wrapper-stack",
        args.hrgym_wrapper_stack,
        "--reward-variant",
        variant,
        "--total-timesteps",
        str(args.total_timesteps),
        "--horizon",
        str(args.horizon),
        "--learning-rate",
        str(args.learning_rate),
        "--n-steps",
        str(args.n_steps),
        "--batch-size",
        str(args.batch_size),
        "--n-epochs",
        str(args.n_epochs),
        "--checkpoint-freq",
        str(args.checkpoint_freq),
        "--best-model-min-episodes",
        str(args.best_model_min_episodes),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--output-dir",
        str(output_dir),
        "--wandb-mode",
        args.wandb_mode,
        "--wandb-project",
        args.wandb_project,
        "--wandb-group",
        args.wandb_group,
        "--wandb-name",
        run_name,
    ]
    if args.backend == "hrgym" and args.hrgym_shield_type:
        values.extend(["--hrgym-shield-type", args.hrgym_shield_type])
    values.extend(_config_args(args))
    if args.backend == "hrgym" and args.expert_bc_rollouts > 0 and args.expert_bc_epochs > 0:
        values.extend(
            [
                "--expert-bc-rollouts",
                str(args.expert_bc_rollouts),
                "--expert-bc-epochs",
                str(args.expert_bc_epochs),
                "--expert-bc-batch-size",
                str(args.expert_bc_batch_size),
                "--expert-bc-learning-rate",
                str(args.expert_bc_learning_rate),
                "--expert-bc-success-only" if args.expert_bc_success_only else "--no-expert-bc-success-only",
                "--expert-bc-fallback-to-failed" if args.expert_bc_fallback_to_failed else "--no-expert-bc-fallback-to-failed",
                "--expert-bc-motion-loss-weight",
                str(args.expert_bc_motion_loss_weight),
                "--expert-bc-gripper-loss-weight",
                str(args.expert_bc_gripper_loss_weight),
                "--expert-bc-speech-policy",
                args.expert_bc_speech_policy,
                "--expert-bc-speech-loss-weight",
                str(args.expert_bc_speech_loss_weight),
                "--expert-dagger-iterations",
                str(args.expert_dagger_iterations),
                "--expert-dagger-rollouts",
                str(args.expert_dagger_rollouts),
                "--expert-dagger-epochs",
                str(args.expert_dagger_epochs),
                "--expert-dagger-max-steps",
                str(args.expert_dagger_max_steps),
            ]
        )
        if args.expert_bc_action_std is not None:
            values.extend(["--expert-bc-action-std", str(args.expert_bc_action_std)])
    return values


def _eval_argv(args: argparse.Namespace, variant: str, model_path: Path, output_dir: Path) -> list[str]:
    values = [
        "--backend",
        args.backend,
        "--handover-env",
        args.handover_env,
        "--hrgym-wrapper-stack",
        args.hrgym_wrapper_stack,
        "--reward-variant",
        variant,
        "--model-path",
        str(model_path),
        "--episodes",
        str(args.eval_episodes),
        "--horizon",
        str(args.horizon),
        "--device",
        args.device,
        "--seed",
        str(args.seed + 10_000),
        "--output-dir",
        str(output_dir),
    ]
    if args.backend == "hrgym" and args.hrgym_shield_type:
        values.extend(["--hrgym-shield-type", args.hrgym_shield_type])
    values.extend(_config_args(args))
    return values


def _config_args(args: argparse.Namespace) -> list[str]:
    values: list[str] = []
    if args.reward_config:
        values.extend(["--reward-config", args.reward_config])
    if args.allostatic_load_config:
        values.extend(["--allostatic-load-config", args.allostatic_load_config])
    if args.human_fsm_config:
        values.extend(["--human-fsm-config", args.human_fsm_config])
    return values


def _summarize_eval_csv(variant: str, path: Path) -> dict[str, Any]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    numeric: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if value == "":
                continue
            try:
                numeric.setdefault(key, []).append(float(value))
            except ValueError:
                pass
    summary = {"reward_variant": variant, "episodes": len(rows)}
    for key, values in numeric.items():
        if values:
            summary[f"mean_{key}"] = sum(values) / len(values)
    return summary


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
