"""Convenience entry point for allostatic PPO."""

import sys

from allostatic_handover.experiments.train_ppo import main


if __name__ == "__main__":
    raise SystemExit(main(["--reward-variant", "allostatic", *sys.argv[1:]]))
