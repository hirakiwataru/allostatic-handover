#!/usr/bin/env python3
"""Check dependencies needed by the exact DreamerV3 integration."""

from __future__ import annotations

import argparse

from allostatic_handover.dreamerv3_exact.dependencies import (
  DEFAULT_DREAMERV3_PATH,
  check_dreamerv3_dependencies,
)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dreamerv3-path", default=DEFAULT_DREAMERV3_PATH)
  parser.add_argument("--fail", action="store_true")
  args = parser.parse_args()

  status = check_dreamerv3_dependencies(args.dreamerv3_path)
  print(status.to_json())
  if args.fail and not status.ok:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
