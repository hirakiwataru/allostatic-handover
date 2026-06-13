#!/usr/bin/env python3
"""Print and optionally render Mjlab Full handover layout diagnostics."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


TASK_BUILDERS = {
  "Mjlab-Allostatic-Handover-Full": "allostatic_handover_full_yam_env_cfg",
  "Mjlab-Allostatic-Handover-Full-TaskOnly": "allostatic_handover_full_task_only_yam_env_cfg",
  "Mjlab-Allostatic-Handover-Full-TaskOnly-GraspedStart": (
    "allostatic_handover_full_task_only_grasped_start_yam_env_cfg"
  ),
}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "task_id",
    choices=tuple(TASK_BUILDERS),
    default="Mjlab-Allostatic-Handover-Full-TaskOnly",
    nargs="?",
  )
  parser.add_argument("--steps", type=int, default=0)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--render-path", type=Path, default=None)
  args = parser.parse_args()

  os.environ.setdefault("MUJOCO_GL", "egl")
  os.environ.setdefault("XDG_CACHE_HOME", "/mnt/k_iwamoto/sim_data/tmp/xdg_cache")
  os.environ.setdefault("MPLCONFIGDIR", "/mnt/k_iwamoto/sim_data/tmp/matplotlib")

  from PIL import Image

  from allostatic_handover.mjlab_tasks import env_cfg as env_cfg_module
  from allostatic_handover.mjlab_tasks.env_cfg import FULL_TABLE_TOP_Z
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

  builder = getattr(env_cfg_module, TASK_BUILDERS[args.task_id])
  cfg = builder(play=True)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(
    cfg=cfg,
    device=args.device,
    render_mode="rgb_array" if args.render_path else None,
  )
  try:
    env.reset()
    robot = env.scene.entities["robot"]
    table = env.scene.entities["table"]
    obj = env.scene.entities["manipulation_object"]
    grasp_id = robot.site_names.index("grasp_site")
    table_top_id = table.site_names.index("table_top")

    _print_state(args.task_id, "reset", env, robot, table, obj, grasp_id, table_top_id)
    if args.steps > 0:
      action = torch.zeros((1, env.action_manager.total_action_dim), device=env.device)
      for _ in range(args.steps):
        env.step(action)
      _print_state(
        args.task_id,
        f"step_{args.steps}",
        env,
        robot,
        table,
        obj,
        grasp_id,
        table_top_id,
      )

    print(f"env_cfg_file={env_cfg_module.__file__}")
    print(f"table_top_reference={FULL_TABLE_TOP_Z:.6f}")
    print(f"action_dim={env.action_manager.total_action_dim}")

    if args.render_path:
      frame = env.render()
      if frame is None:
        raise RuntimeError("env.render() returned None")
      if frame.ndim == 4:
        frame = frame[0]
      args.render_path.parent.mkdir(parents=True, exist_ok=True)
      Image.fromarray(frame).save(args.render_path)
      print(f"render_path={args.render_path}")
  finally:
    env.close()


def _print_state(
  task_id: str,
  label: str,
  env,
  robot,
  table,
  obj,
  grasp_id: int,
  table_top_id: int,
) -> None:
  robot_root = robot.data.root_link_pos_w[0].detach().cpu().tolist()
  grasp_site = robot.data.site_pos_w[0, grasp_id].detach().cpu().tolist()
  table_top = table.data.site_pos_w[0, table_top_id].detach().cpu().tolist()
  object_root = obj.data.root_link_pos_w[0].detach().cpu().tolist()
  object_quat = obj.data.root_link_quat_w[0].detach().cpu().tolist()
  object_to_root_xy = torch.norm(
    obj.data.root_link_pos_w[0, :2] - robot.data.root_link_pos_w[0, :2]
  )
  object_to_grasp_xy = torch.norm(
    obj.data.root_link_pos_w[0, :2] - robot.data.site_pos_w[0, grasp_id, :2]
  )
  print(f"task_id={task_id}")
  print(f"{label}.robot_root={_fmt_vec(robot_root)}")
  print(f"{label}.grasp_site={_fmt_vec(grasp_site)}")
  print(f"{label}.table_top={_fmt_vec(table_top)}")
  print(f"{label}.object_root={_fmt_vec(object_root)}")
  print(f"{label}.object_quat_wxyz={_fmt_vec(object_quat)}")
  print(f"{label}.object_to_root_xy={float(object_to_root_xy):.6f}")
  print(f"{label}.object_to_grasp_xy={float(object_to_grasp_xy):.6f}")


def _fmt_vec(values: list[float]) -> str:
  return "(" + ", ".join(f"{value:.6f}" for value in values) + ")"


if __name__ == "__main__":
  main()
