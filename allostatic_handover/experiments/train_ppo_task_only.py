"""Convenience entry point for task-only PPO."""

import sys

from allostatic_handover.experiments.train_ppo import main


if __name__ == "__main__":
    raise SystemExit(main(["--reward-variant", "task_only", *sys.argv[1:]]))
