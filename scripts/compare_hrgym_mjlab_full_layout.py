#!/usr/bin/env python3
"""Compare HRGym RobotHumanHandoverCart and Mjlab Full layout coordinates."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECTS_ROOT = REPO_ROOT.parent
HRGYM_ROOT = PROJECTS_ROOT / "human-robot-gym"
MJLAB_ROOT = PROJECTS_ROOT / "mjlab"
DEFAULT_HRGYM_PYTHON = REPO_ROOT / ".conda/bin/python"
DEFAULT_MJLAB_PYTHON = MJLAB_ROOT / ".venv/bin/python"
SCRIPT_PATH = Path(__file__).resolve()
ANIMATION_NAMES = tuple(f"RobotHumanHandover/{idx}" for idx in range(9))


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--side", choices=("both", "hrgym", "mjlab"), default="both")
  parser.add_argument("--output", type=Path, default=None)
  parser.add_argument("--seed", type=int, default=101)
  parser.add_argument("--hrgym-python", type=Path, default=DEFAULT_HRGYM_PYTHON)
  parser.add_argument("--mjlab-python", type=Path, default=DEFAULT_MJLAB_PYTHON)
  args = parser.parse_args()

  if args.side == "hrgym":
    _emit(_collect_hrgym(seed=args.seed), args.output)
    return
  if args.side == "mjlab":
    _emit(_collect_mjlab(), args.output)
    return

  hrgym = _run_child(args.hrgym_python, "hrgym", args.seed)
  mjlab = _run_child(args.mjlab_python, "mjlab", args.seed)
  result = {
    "hrgym": hrgym,
    "mjlab": mjlab,
    "comparison": _compare(hrgym, mjlab),
  }
  _emit(result, args.output)


def _emit(data: dict[str, Any], output: Path | None) -> None:
  text = json.dumps(data, indent=2, sort_keys=True)
  if output is None:
    print(text)
    return
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(text + "\n", encoding="utf-8")


def _run_child(python: Path, side: str, seed: int) -> dict[str, Any]:
  if not python.exists():
    raise FileNotFoundError(f"Missing Python for {side}: {python}")
  with tempfile.TemporaryDirectory(prefix=f"layout-{side}-") as tmp:
    output = Path(tmp) / f"{side}.json"
    env = _child_env()
    cmd = [
      str(python),
      str(SCRIPT_PATH),
      "--side",
      side,
      "--seed",
      str(seed),
      "--output",
      str(output),
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
      sys.stderr.write(proc.stdout)
      sys.stderr.write(proc.stderr)
      raise subprocess.CalledProcessError(proc.returncode, cmd)
    return json.loads(output.read_text(encoding="utf-8"))


def _child_env() -> dict[str, str]:
  env = os.environ.copy()
  pythonpath = [
    str(REPO_ROOT),
    str(HRGYM_ROOT),
    str(MJLAB_ROOT / "src"),
  ]
  if env.get("PYTHONPATH"):
    pythonpath.append(env["PYTHONPATH"])
  env["PYTHONPATH"] = os.pathsep.join(pythonpath)
  env.setdefault("MUJOCO_GL", "egl")
  env.setdefault("XDG_CACHE_HOME", "/mnt/k_iwamoto/sim_data/tmp/xdg_cache")
  env.setdefault("MPLCONFIGDIR", "/mnt/k_iwamoto/sim_data/tmp/matplotlib")
  return env


def _collect_hrgym(seed: int) -> dict[str, Any]:
  from allostatic_handover.experiments.env_factory import make_env, reset_env
  from allostatic_handover.experiments.hrgym_exact_rhh_sac import (
    _apply_runtime_compatibility_patches,
  )

  _apply_runtime_compatibility_patches()
  records = []
  object_bin_boundaries = None
  table_full_size = None
  table_offset = None
  for animation_id, animation_name in enumerate(ANIMATION_NAMES):
    env = make_env(
      backend="hrgym",
      handover_env="original",
      horizon=1000,
      seed=seed,
      render=False,
      human_animation_names=[animation_name],
      human_animation_freq=90,
      human_rand=[0.0, 0.0, 0.0],
      n_animations_sampled_per_100_steps=1,
    )
    try:
      reset_env(env)
      raw = env
      while hasattr(raw, "env"):
        raw = raw.env
      sim = raw.sim
      if object_bin_boundaries is None:
        object_bin_boundaries = list(raw._get_default_object_bin_boundaries())
        table_full_size = _list(raw.table_full_size)
        table_offset = _list(raw.table_offset)
      records.append(
        {
          "animation_id": animation_id,
          "animation_name": animation_name,
          "holding_hand": raw.object_holding_hand,
          "frame": int(raw.animation_time),
          "target": _list(raw.target_pos),
          "object_grip": _list(sim.data.get_body_xpos("manipulation_object_grip")),
          "robot_base": _list(sim.data.get_body_xpos("robot0_base")),
          "robot_right_hand": _list(sim.data.get_body_xpos("robot0_right_hand")),
          "human_l_hand_site": _list(sim.data.get_site_xpos("Human_L_Hand")),
          "human_r_hand_site": _list(sim.data.get_site_xpos("Human_R_Hand")),
        }
      )
    finally:
      env.close()

  return {
    "backend": "hrgym",
    "animation_names": list(ANIMATION_NAMES),
    "human_animation_freq": 90,
    "human_rand": [0.0, 0.0, 0.0],
    "object_bin_boundaries": object_bin_boundaries,
    "table_full_size": table_full_size,
    "table_offset": table_offset,
    "animations": records,
  }


def _collect_mjlab() -> dict[str, Any]:
  import torch

  from allostatic_handover.mjlab_tasks.env_cfg import (
    FULL_HRGYM_OBJECT_Z,
    FULL_YAM_ROOT_POS,
    HRGYM_REFERENCE_ROBOT_HAND_POS,
    allostatic_handover_full_yam_env_cfg,
  )
  from mjlab.envs import ManagerBasedRlEnv

  cfg = allostatic_handover_full_yam_env_cfg(play=True)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
  try:
    env.reset()
    command = env.command_manager.get_term("handover")
    human = env.scene["human"]
    robot = env.scene["robot"]
    obj = env.scene["manipulation_object"]
    table = env.scene["table"]
    grasp_id = robot.site_names.index("grasp_site")
    left_site_id = human.site_names.index("L_Hand")
    right_site_id = human.site_names.index("R_Hand")
    table_top_id = table.site_names.index("table_top")

    records = []
    for animation_id, animation in enumerate(command.animation_library.animations):
      command.animation_id[:] = animation_id
      command.animation_frame[:] = 0.0
      command._classic_animation_frame[:] = 0.0
      command._delayed_animation_frames[:] = 0.0
      command._write_human_animation_state()
      env.sim.forward()
      env.scene.update(0.0)
      target = command._read_palm_target()[0]
      records.append(
        {
          "animation_id": animation_id,
          "animation_name": command.cfg.animation_names[animation_id],
          "holding_hand": animation.object_holding_hand,
          "frame": 0,
          "target": _tensor_list(target),
          "human_l_hand_site": _tensor_list(human.data.site_pos_w[0, left_site_id]),
          "human_r_hand_site": _tensor_list(human.data.site_pos_w[0, right_site_id]),
        }
      )

    pose_range = command.cfg.object_pose_range
    return {
      "backend": "mjlab",
      "animation_names": list(command.cfg.animation_names),
      "human_animation_freq": command.cfg.human_animation_freq,
      "object_pose_range": {
        "x": list(pose_range.x),
        "y": list(pose_range.y),
        "z": list(pose_range.z),
        "yaw": list(pose_range.yaw),
      },
      "reference_object_z": FULL_HRGYM_OBJECT_Z,
      "reference_robot_hand": list(HRGYM_REFERENCE_ROBOT_HAND_POS),
      "configured_yam_root": list(FULL_YAM_ROOT_POS),
      "robot_root": _tensor_list(robot.data.root_link_pos_w[0]),
      "grasp_site": _tensor_list(robot.data.site_pos_w[0, grasp_id]),
      "object_root": _tensor_list(obj.data.root_link_pos_w[0]),
      "table_top": _tensor_list(table.data.site_pos_w[0, table_top_id]),
      "animations": records,
    }
  finally:
    env.close()


def _compare(hrgym: dict[str, Any], mjlab: dict[str, Any]) -> dict[str, Any]:
  target_errors = []
  for hrgym_record, mjlab_record in zip(hrgym["animations"], mjlab["animations"]):
    target_errors.append(
      {
        "animation_id": hrgym_record["animation_id"],
        "target_linf_error": _linf(hrgym_record["target"], mjlab_record["target"]),
      }
    )
  reference_hand = hrgym["animations"][0]["robot_right_hand"]
  return {
    "max_target_linf_error": max(item["target_linf_error"] for item in target_errors),
    "target_errors": target_errors,
    "grasp_site_to_hrgym_right_hand_linf_error": _linf(
      reference_hand,
      mjlab["grasp_site"],
    ),
    "object_z_error_from_hrgym_reference": abs(
      mjlab["object_pose_range"]["z"][0] - hrgym["animations"][0]["object_grip"][2]
    ),
  }


def _linf(left: list[float], right: list[float]) -> float:
  return max(abs(a - b) for a, b in zip(left, right))


def _list(values: Any) -> list[float]:
  return [float(value) for value in values]


def _tensor_list(values: Any) -> list[float]:
  return [float(value) for value in values.detach().cpu()]


if __name__ == "__main__":
  main()
