"""Create simple SVG summaries from episodes.csv without extra dependencies."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes_csv")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    rows = _read_rows(Path(args.episodes_csv))
    output = Path(args.output or Path(args.episodes_csv).with_suffix(".svg"))
    output.write_text(_svg(rows), encoding="utf-8")
    print(f"wrote {output.resolve()}")
    return 0


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _svg(rows: list[dict[str, str]]) -> str:
    width, height = 920, 420
    plot_w, plot_h = 760, 280
    x0, y0 = 110, 70
    loads = [_float(row.get("allostatic_load_total")) for row in rows]
    successes = [_float(row.get("success")) for row in rows]
    max_load = max(loads + [1.0])
    bars = []
    n = max(1, len(rows))
    bar_w = max(6, plot_w / n * 0.7)
    for i, row in enumerate(rows):
        x = x0 + i * plot_w / n
        load_h = loads[i] / max_load * plot_h
        color = "#059669" if successes[i] > 0.5 else "#dc2626"
        bars.append(
            f'<rect x="{x:.1f}" y="{y0 + plot_h - load_h:.1f}" width="{bar_w:.1f}" '
            f'height="{load_h:.1f}" fill="{color}" opacity="0.82"><title>'
            f'episode {row.get("episode_id", i)} load {loads[i]:.3f}</title></rect>'
        )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="32" y="34" font-family="sans-serif" font-size="20" font-weight="700">Allostatic Load By Episode</text>
<line x1="{x0}" y1="{y0 + plot_h}" x2="{x0 + plot_w}" y2="{y0 + plot_h}" stroke="#1f2933"/>
<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + plot_h}" stroke="#1f2933"/>
<text x="30" y="{y0 + 10}" font-family="sans-serif" font-size="12">load</text>
<text x="{x0 + plot_w - 20}" y="{y0 + plot_h + 38}" font-family="sans-serif" font-size="12">episode</text>
{''.join(bars)}
<text x="{x0}" y="{height - 36}" font-family="sans-serif" font-size="12" fill="#059669">green: success</text>
<text x="{x0 + 120}" y="{height - 36}" font-family="sans-serif" font-size="12" fill="#dc2626">red: no success</text>
</svg>
"""


def _float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    raise SystemExit(main())
