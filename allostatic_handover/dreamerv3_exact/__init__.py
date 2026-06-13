"""Exact DreamerV3 integration helpers.

This package intentionally imports DreamerV3/JAX dependencies lazily so the
rest of allostatic-handover remains usable in the Mjlab PPO environment.
"""

from .dependencies import check_dreamerv3_dependencies

__all__ = ["check_dreamerv3_dependencies"]
