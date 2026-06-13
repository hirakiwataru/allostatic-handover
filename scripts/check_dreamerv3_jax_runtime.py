#!/usr/bin/env python3
"""Run a minimal JAX runtime check for the exact DreamerV3 environment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from allostatic_handover.dreamerv3_exact.dependencies import (
  DEFAULT_DREAMERV3_PATH,
)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dreamerv3-path", default=DEFAULT_DREAMERV3_PATH)
  parser.add_argument("--platform", default="cpu", choices=("cpu", "cuda"))
  parser.add_argument("--fail", action="store_true")
  args = parser.parse_args()

  os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
  os.environ["JAX_PLATFORM_NAME"] = args.platform
  sys.path.insert(0, args.dreamerv3_path)
  sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

  payload: dict[str, object] = {
    "dreamerv3_path": args.dreamerv3_path,
    "platform": args.platform,
    "ok": False,
  }
  try:
    import jax
    import jax.numpy as jnp

    devices = jax.devices(args.platform)
    x = jnp.ones((16, 16), dtype=jnp.float32)
    y = (x @ x).block_until_ready()
    payload.update(
      {
        "ok": True,
        "jax_version": jax.__version__,
        "devices": [str(device) for device in devices],
        "matmul_00": float(y[0, 0]),
      }
    )
  except Exception as exc:
    payload.update(
      {
        "ok": False,
        "error": f"{type(exc).__name__}: {exc}",
      }
    )

  print(json.dumps(payload, indent=2, sort_keys=True))
  if args.fail and not payload["ok"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
