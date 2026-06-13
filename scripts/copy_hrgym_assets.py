#!/usr/bin/env python3
"""Copy selected human-robot-gym handover assets for local Mjlab experiments.

The copied vendor assets are intentionally ignored by git until their license
status is confirmed.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_HRGYM_ROOT = Path("/mnt/k_iwamoto/sim_data/Projects/human-robot-gym")
DEFAULT_DEST = Path(__file__).resolve().parents[1] / "assets/vendor/human_robot_gym"
ANIMATION_REL = Path(
  "human_robot_gym/models/assets/human/animations/"
  "human-robot-animations/RobotHumanHandover"
)
ASSETS_REL = Path("human_robot_gym/models/assets")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--hrgym-root",
    type=Path,
    default=DEFAULT_HRGYM_ROOT,
    help="Path to the human-robot-gym checkout.",
  )
  parser.add_argument(
    "--dest",
    type=Path,
    default=DEFAULT_DEST,
    help="Destination vendor asset root inside allostatic-handover.",
  )
  parser.add_argument(
    "--include-animation-files",
    action="store_true",
    help="Also copy .bvh/.csv/.pkl files. By default only *_info.json is copied.",
  )
  parser.add_argument(
    "--include-full-handover-assets",
    action="store_true",
    help=(
      "Copy HRGym human XML/meshes/textures, table XML, and handover pkl/info "
      "files required by Mjlab-Allostatic-Handover-Full."
    ),
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  src = args.hrgym_root / ANIMATION_REL
  if not src.exists():
    raise FileNotFoundError(f"RobotHumanHandover asset directory not found: {src}")

  dest = args.dest / "human/animations/human-robot-animations/RobotHumanHandover"
  dest.mkdir(parents=True, exist_ok=True)

  patterns = ["*_info.json"]
  if args.include_animation_files or args.include_full_handover_assets:
    patterns.extend(["*.bvh", "*.csv", "*.pkl"])

  copied = 0
  for pattern in patterns:
    for file in sorted(src.glob(pattern)):
      shutil.copy2(file, dest / file.name)
      copied += 1

  if args.include_full_handover_assets:
    assets_src = args.hrgym_root / ASSETS_REL
    if not assets_src.exists():
      raise FileNotFoundError(f"human-robot-gym asset root not found: {assets_src}")

    full_asset_pairs = [
      (assets_src / "human/human.xml", args.dest / "human/human.xml"),
      (assets_src / "arenas/table_arena.xml", args.dest / "arenas/table_arena.xml"),
    ]
    for file_src, file_dest in full_asset_pairs:
      file_dest.parent.mkdir(parents=True, exist_ok=True)
      shutil.copy2(file_src, file_dest)
      copied += 1

    for dir_name in ("human/meshes",):
      dir_src = assets_src / dir_name
      dir_dest = args.dest / dir_name
      if dir_dest.exists():
        shutil.rmtree(dir_dest)
      shutil.copytree(dir_src, dir_dest)
      copied += sum(1 for p in dir_dest.rglob("*") if p.is_file())

    textures_dest = args.dest / "textures"
    textures_dest.mkdir(parents=True, exist_ok=True)
    for texture_name in ("skin.png", "jeans.png", "green-shirt.png"):
      shutil.copy2(assets_src / "textures" / texture_name, textures_dest / texture_name)
      copied += 1

  print(f"Copied {copied} files from {args.hrgym_root} to {args.dest}")


if __name__ == "__main__":
  main()
