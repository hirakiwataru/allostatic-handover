"""Dependency checks for the exact DreamerV3 path."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import importlib
import json
import sys

DEFAULT_DREAMERV3_PATH = "/mnt/k_iwamoto/sim_data/Projects/dreamerv3"


@dataclass
class DreamerV3DependencyStatus:
  dreamerv3_path: str
  repo_present: bool
  missing: list[str]
  import_errors: dict[str, str]

  @property
  def ok(self) -> bool:
    return self.repo_present and not self.missing

  def to_json(self) -> str:
    return json.dumps(asdict(self), indent=2, sort_keys=True)


def check_dreamerv3_dependencies(
  dreamerv3_path: str = DEFAULT_DREAMERV3_PATH,
  modules: Iterable[str] | None = None,
) -> DreamerV3DependencyStatus:
  """Check whether exact DreamerV3 training dependencies are importable."""
  path = Path(dreamerv3_path)
  if path.exists() and str(path) not in sys.path:
    sys.path.insert(0, str(path))
  modules = tuple(
    modules
    or (
      "jax",
      "ninjax",
      "optax",
      "einops",
      "elements",
      "embodied",
      "dreamerv3.rssm",
      "dreamerv3.agent",
    )
  )
  missing: list[str] = []
  import_errors: dict[str, str] = {}
  for module in modules:
    try:
      importlib.import_module(module)
    except Exception as exc:
      missing.append(module)
      import_errors[module] = f"{type(exc).__name__}: {exc}"
  return DreamerV3DependencyStatus(
    dreamerv3_path=str(path),
    repo_present=path.exists(),
    missing=missing,
    import_errors=import_errors,
  )
