#!/usr/bin/env python3
"""Render random HRGym human animations in the Mjlab Full handover scene."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--samples", type=int, default=6)
  parser.add_argument("--seed", type=int, default=17)
  parser.add_argument("--reach-progress", type=float, default=0.55)
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path(
      "/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/"
      "outputs/visual_checks/mjlab_full_random_animations"
    ),
  )
  args = parser.parse_args()

  os.environ.setdefault("MUJOCO_GL", "egl")
  os.environ.setdefault("XDG_CACHE_HOME", "/mnt/k_iwamoto/sim_data/tmp/xdg_cache")
  os.environ.setdefault("MPLCONFIGDIR", "/mnt/k_iwamoto/sim_data/tmp/matplotlib")

  from allostatic_handover.mjlab_tasks.env_cfg import allostatic_handover_full_yam_env_cfg
  from allostatic_handover.mjlab_tasks.mdp import HandoverPhase
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

  rng = random.Random(args.seed)
  args.output_dir.mkdir(parents=True, exist_ok=True)

  cfg = allostatic_handover_full_yam_env_cfg(play=True)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode="rgb_array")
  frames: list[Image.Image] = []
  sampled_ids: list[int] = []
  try:
    env.reset()
    command = env.command_manager.get_term("handover")
    num_animations = len(command.animation_library)
    for sample_idx in range(args.samples):
      animation_id = rng.randrange(num_animations)
      sampled_ids.append(animation_id)
      command.animation_id[:] = animation_id
      command.human_readiness[:] = 1.0
      command.phase[:] = int(HandoverPhase.REACH_OUT)
      key0, key1, _ = command._current_keyframe_tensors()
      frame = key0 + torch.clamp(
        torch.tensor(args.reach_progress, device=env.device),
        0.0,
        1.0,
      ) * (key1 - key0)
      command.animation_frame[:] = frame
      command._classic_animation_frame[:] = frame
      command._delayed_animation_frames[:] = 0.0
      command._last_update_step = -1
      command.pre_reward_update()
      env.scene.write_data_to_sim()
      env.sim.forward()
      env.scene.update(0.0)

      rendered = env.render()
      if rendered is None:
        raise RuntimeError("env.render() returned None")
      if rendered.ndim == 4:
        rendered = rendered[0]
      image = Image.fromarray(np.asarray(rendered, dtype=np.uint8))
      label = f"animation={animation_id} sample={sample_idx}"
      draw = ImageDraw.Draw(image)
      draw.rectangle((8, 8, 260, 36), fill=(0, 0, 0))
      draw.text((14, 14), label, fill=(255, 255, 255))
      path = args.output_dir / f"sample_{sample_idx:02d}_animation_{animation_id}.png"
      image.save(path)
      frames.append(image)
      print(f"saved={path} {label}")
  finally:
    env.close()

  if frames:
    montage = _make_montage(frames)
    montage_path = args.output_dir / "montage.png"
    montage.save(montage_path)
    print(f"montage={montage_path}")
    print("sampled_animation_ids=" + ",".join(str(idx) for idx in sampled_ids))


def _make_montage(frames: list[Image.Image], cols: int = 3) -> Image.Image:
  width, height = frames[0].size
  rows = (len(frames) + cols - 1) // cols
  montage = Image.new("RGB", (cols * width, rows * height), (0, 0, 0))
  for idx, image in enumerate(frames):
    row, col = divmod(idx, cols)
    montage.paste(image, (col * width, row * height))
  return montage


if __name__ == "__main__":
  main()
