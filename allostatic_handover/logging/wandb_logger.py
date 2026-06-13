"""Optional Weights & Biases logging."""

from __future__ import annotations

from typing import Any, Mapping


class WandbRun:
    """Small wrapper that degrades cleanly when wandb is unavailable/disabled."""

    def __init__(
        self,
        enabled: bool = False,
        project: str = "allostatic-handover-mvp",
        entity: str | None = None,
        group: str | None = None,
        mode: str = "disabled",
        config: Mapping[str, Any] | None = None,
        tags: list[str] | None = None,
        name: str | None = None,
    ):
        self.enabled = enabled and mode != "disabled"
        self._run = None
        self._wandb = None
        if not self.enabled:
            return
        try:
            import wandb
        except ImportError:
            print("[wandb] package is not installed; continuing with local logging only.")
            self.enabled = False
            return
        self._wandb = wandb
        self._run = wandb.init(
            project=project,
            entity=entity,
            group=group,
            mode=mode,
            config=dict(config or {}),
            tags=tags,
            name=name,
        )

    def log(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        if self.enabled and self._wandb is not None:
            self._wandb.log(dict(metrics), step=step)

    def finish(self) -> None:
        if self.enabled and self._run is not None:
            self._run.finish()
