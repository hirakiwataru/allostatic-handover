"""Run the minimal vs excessive scripted comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from allostatic_handover.experiments.run_scripted_rollouts import main as run_scripted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["mock", "hrgym"], default="mock")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=160)
    parser.add_argument("--output-root", default="outputs/degeneracy_eval")
    args = parser.parse_args(argv)

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    for policy in ["minimal_speech", "excessive_speech", "human_waiting", "allostatic_aware"]:
        for reward_variant in ["task_only", "allostatic"]:
            run_scripted(
                [
                    "--backend",
                    args.backend,
                    "--policy",
                    policy,
                    "--reward-variant",
                    reward_variant,
                    "--episodes",
                    str(args.episodes),
                    "--horizon",
                    str(args.horizon),
                    "--output-dir",
                    str(root / f"{args.backend}_{policy}_{reward_variant}"),
                ]
            )
    print(f"comparison logs written under {root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
