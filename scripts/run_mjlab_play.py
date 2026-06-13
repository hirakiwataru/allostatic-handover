"""Run Mjlab native viewer with reliable Ctrl+C cleanup."""

from __future__ import annotations

import argparse
import atexit
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path("/mnt/k_iwamoto/sim_data/Projects/allostatic-handover")
MJLAB_ROOT = Path("/mnt/k_iwamoto/sim_data/Projects/mjlab")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("task_id", help="Mjlab task id to open with uv run play.")
  parser.add_argument("--agent", default="zero")
  parser.add_argument("--viewer", default="native")
  parser.add_argument("--checkpoint-file", default="")
  parser.add_argument(
      "--no-terminations",
      nargs="?",
      const="True",
      default=None,
      help="Pass through to Mjlab play as '--no-terminations True'.",
  )
  parser.add_argument("--display", default=os.environ.get("MJLAB_DISPLAY", ":1"))
  parser.add_argument(
      "--xauthority",
      default=os.environ.get(
          "MJLAB_XAUTHORITY",
          f"/run/user/{os.getuid()}/gdm/Xauthority",
      ),
  )
  parser.add_argument("--dry-run", action="store_true")
  parser.add_argument("--live-dashboard", action="store_true")
  parser.add_argument("--live-log", default="")
  parser.add_argument("--live-log-interval", type=int, default=10)
  parser.add_argument("--dashboard-host", default="0.0.0.0")
  parser.add_argument("--dashboard-port", type=int, default=7860)
  return parser.parse_known_args()


def build_env(args: argparse.Namespace) -> dict[str, str]:
  env = os.environ.copy()
  env.setdefault("UV_CACHE_DIR", str(PROJECT_ROOT / ".uvcache"))
  env.setdefault("XDG_CACHE_HOME", "/mnt/k_iwamoto/sim_data/tmp/xdg_cache")
  env.setdefault("MPLCONFIGDIR", "/mnt/k_iwamoto/sim_data/tmp/matplotlib")
  env["DISPLAY"] = args.display
  env["XAUTHORITY"] = args.xauthority
  env["MUJOCO_GL"] = "glfw"
  live_log = resolve_live_log_path(args)
  if live_log is not None:
    env["ALLOSTATIC_MJLAB_LIVE_LOG"] = str(live_log)
    env["ALLOSTATIC_MJLAB_LIVE_LOG_INTERVAL"] = str(max(1, args.live_log_interval))
  return env


def resolve_live_log_path(args: argparse.Namespace) -> Path | None:
  if not (args.live_dashboard or args.live_log):
    return None
  if args.live_log:
    return Path(args.live_log)
  stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  task_name = args.task_id.replace("/", "_")
  agent_name = args.agent
  return PROJECT_ROOT / "outputs" / "mjlab_live" / f"{task_name}_{agent_name}_{stamp}" / "live.jsonl"


def start_dashboard(args: argparse.Namespace) -> subprocess.Popen[bytes] | None:
  if not args.live_dashboard:
    return None
  cmd = [
      sys.executable,
      "-m",
      "allostatic_handover.dashboard.app",
      "--log-dir",
      str(PROJECT_ROOT / "outputs"),
      "--host",
      args.dashboard_host,
      "--port",
      str(args.dashboard_port),
  ]
  proc = subprocess.Popen(
      cmd,
      cwd=PROJECT_ROOT,
      env=os.environ.copy(),
      start_new_session=True,
  )
  print(
      "live dashboard:",
      f"http://127.0.0.1:{args.dashboard_port}/live.html",
      flush=True,
  )
  return proc


def build_cmd(args: argparse.Namespace, passthrough: list[str]) -> list[str]:
  cmd = [
      "uv",
      "run",
      "play",
      args.task_id,
      "--agent",
      args.agent,
      "--viewer",
      args.viewer,
  ]
  if args.checkpoint_file:
    cmd.extend(["--checkpoint-file", args.checkpoint_file])
  if args.no_terminations is not None:
    value = str(args.no_terminations).lower()
    enabled = value not in {"0", "false", "no", "off"}
    cmd.extend(["--no-terminations", "True" if enabled else "False"])
  cmd.extend(passthrough)
  return cmd


def terminate_process_group(proc: subprocess.Popen[bytes], timeout_s: float = 5.0) -> None:
  if proc.poll() is not None:
    return

  try:
    os.killpg(proc.pid, signal.SIGTERM)
  except ProcessLookupError:
    return

  deadline = time.monotonic() + timeout_s
  while time.monotonic() < deadline:
    if proc.poll() is not None:
      return
    time.sleep(0.1)

  if proc.poll() is None:
    try:
      os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
      pass


def main() -> int:
  args, passthrough = parse_args()
  cmd = build_cmd(args, passthrough)
  env = build_env(args)

  if args.dry_run:
    print("cwd:", MJLAB_ROOT)
    print("env DISPLAY:", env["DISPLAY"])
    print("env XAUTHORITY:", env["XAUTHORITY"])
    print("env MUJOCO_GL:", env["MUJOCO_GL"])
    print("env ALLOSTATIC_MJLAB_LIVE_LOG:", env.get("ALLOSTATIC_MJLAB_LIVE_LOG", ""))
    print("cmd:", " ".join(cmd))
    return 0

  dashboard_proc = start_dashboard(args)
  proc = subprocess.Popen(
      cmd,
      cwd=MJLAB_ROOT,
      env=env,
      start_new_session=True,
  )

  def cleanup() -> None:
    terminate_process_group(proc)
    if dashboard_proc is not None:
      terminate_process_group(dashboard_proc)

  atexit.register(cleanup)

  def handle_stop(signum: int, _frame: object) -> None:
    print(f"\nStopping Mjlab viewer process group after signal {signum}...", flush=True)
    cleanup()
    raise SystemExit(128 + signum)

  signal.signal(signal.SIGINT, handle_stop)
  signal.signal(signal.SIGTERM, handle_stop)

  try:
    return proc.wait()
  except KeyboardInterrupt:
    cleanup()
    return 130


if __name__ == "__main__":
  raise SystemExit(main())
