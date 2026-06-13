"""Utilities for local human-robot-gym assets used by Mjlab tasks."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import mujoco
import torch

DEFAULT_VENDOR_ROOT = (
  Path(__file__).resolve().parents[2] / "assets/vendor/human_robot_gym"
)
ROBOT_HUMAN_HANDOVER_REL = Path(
  "human/animations/human-robot-animations/RobotHumanHandover"
)
DEFAULT_FULL_ANIMATION_NAMES = tuple(f"RobotHumanHandover/{i}" for i in range(9))

HRGYM_HUMAN_JOINT_NAMES = (
  "L_Hip_z",
  "L_Hip_y",
  "L_Hip_x",
  "L_Knee_z",
  "L_Knee_y",
  "L_Knee_x",
  "L_Ankle_z",
  "L_Ankle_y",
  "L_Ankle_x",
  "L_Toe_z",
  "L_Toe_y",
  "L_Toe_x",
  "R_Hip_z",
  "R_Hip_y",
  "R_Hip_x",
  "R_Knee_z",
  "R_Knee_y",
  "R_Knee_x",
  "R_Ankle_z",
  "R_Ankle_y",
  "R_Ankle_x",
  "R_Toe_z",
  "R_Toe_y",
  "R_Toe_x",
  "Torso_z",
  "Torso_y",
  "Torso_x",
  "Spine_z",
  "Spine_y",
  "Spine_x",
  "Chest_z",
  "Chest_y",
  "Chest_x",
  "Neck_z",
  "Neck_y",
  "Neck_x",
  "Head_z",
  "Head_y",
  "Head_x",
  "L_Thorax_z",
  "L_Thorax_y",
  "L_Thorax_x",
  "L_Shoulder_z",
  "L_Shoulder_y",
  "L_Shoulder_x",
  "L_Elbow_z",
  "L_Elbow_y",
  "L_Elbow_x",
  "L_Wrist_z",
  "L_Wrist_y",
  "L_Wrist_x",
  "L_Hand_z",
  "L_Hand_y",
  "L_Hand_x",
  "R_Thorax_z",
  "R_Thorax_y",
  "R_Thorax_x",
  "R_Shoulder_z",
  "R_Shoulder_y",
  "R_Shoulder_x",
  "R_Elbow_z",
  "R_Elbow_y",
  "R_Elbow_x",
  "R_Wrist_z",
  "R_Wrist_y",
  "R_Wrist_x",
  "R_Hand_z",
  "R_Hand_y",
  "R_Hand_x",
)


def require_vendor_root(vendor_root: str | Path = DEFAULT_VENDOR_ROOT) -> Path:
  root = Path(vendor_root)
  if root.exists():
    return root
  raise FileNotFoundError(
    "human-robot-gym vendor assets are missing. Run:\n"
    "  cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover\n"
    "  python scripts/copy_hrgym_assets.py --include-full-handover-assets"
  )


def hrgym_human_spec(vendor_root: str | Path = DEFAULT_VENDOR_ROOT) -> mujoco.MjSpec:
  """Load HRGym's human XML in a shape Mjlab can attach as an entity."""
  root = require_vendor_root(vendor_root)
  xml_path = root / "human/human.xml"
  if not xml_path.exists():
    raise FileNotFoundError(
      f"Missing HRGym human XML: {xml_path}\n"
      "Run: python scripts/copy_hrgym_assets.py --include-full-handover-assets"
    )
  spec = mujoco.MjSpec.from_file(str(xml_path))
  for body in spec.bodies:
    if body.name == "object":
      body.mocap = False
      break
  for geom in spec.geoms:
    geom.margin = 0.0
    geom.contype = 0
    geom.conaffinity = 0
  return spec


def hrgym_table_spec() -> mujoco.MjSpec:
  """Create an HRGym-sized table without depending on robosuite XML mutation."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="table", pos=(0.0, 0.0, 0.82))
  body.add_geom(
    name="table_collision",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(0.75, 1.0, 0.025),
    pos=(0.0, 0.0, 0.0),
    rgba=(0.45, 0.33, 0.24, 1.0),
    mass=0.0,
  )
  body.add_site(
    name="table_top",
    pos=(0.0, 0.0, 0.025),
    size=(0.01,),
    rgba=(0.0, 0.0, 0.0, 0.0),
  )
  return spec


def hrgym_hammer_spec() -> mujoco.MjSpec:
  """Create a lightweight hammer-like manipulation object for handover."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="manipulation_object")
  body.add_freejoint(name="manipulation_object_joint")
  body.add_geom(
    name="hammer_handle",
    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
    fromto=(-0.14, 0.0, 0.0, 0.14, 0.0, 0.0),
    size=(0.021,),
    mass=0.05,
    rgba=(0.70, 0.46, 0.23, 1.0),
  )
  body.add_geom(
    name="hammer_head",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(0.15, 0.0, 0.0),
    size=(0.045, 0.035, 0.035),
    mass=0.04,
    rgba=(0.46, 0.48, 0.50, 1.0),
  )
  body.add_site(
    name="manipulation_object_grip",
    pos=(0.0, 0.0, 0.0),
    size=(0.012,),
    rgba=(0.1, 0.8, 1.0, 0.35),
  )
  return spec


@dataclass(frozen=True)
class HrgymAnimation:
  name: str
  info: dict
  root_pos: torch.Tensor
  root_quat_xyzw: torch.Tensor
  joint_pos: torch.Tensor

  @property
  def num_frames(self) -> int:
    return int(self.root_pos.shape[0])

  @property
  def keyframes(self) -> tuple[int, int]:
    first, second = self.info["keyframes"]
    return int(first), int(second)

  @property
  def object_holding_hand(self) -> str:
    return str(self.info.get("object_holding_hand", "right"))


class HrgymAnimationLibrary:
  """Torch-backed loader for HRGym RobotHumanHandover animations."""

  def __init__(
    self,
    vendor_root: str | Path = DEFAULT_VENDOR_ROOT,
    animation_names: tuple[str, ...] = DEFAULT_FULL_ANIMATION_NAMES,
    *,
    device: str | torch.device = "cpu",
  ) -> None:
    root = require_vendor_root(vendor_root)
    anim_root = root / ROBOT_HUMAN_HANDOVER_REL
    if not anim_root.exists():
      raise FileNotFoundError(
        f"Missing HRGym handover animation directory: {anim_root}\n"
        "Run: python scripts/copy_hrgym_assets.py --include-full-handover-assets"
      )
    self.vendor_root = root
    self.animation_names = animation_names
    self.device = torch.device(device)
    self.animations = tuple(
      self._load_one(anim_root, name).to_device(self.device)
      for name in animation_names
    )
    if not self.animations:
      raise ValueError("At least one HRGym handover animation is required.")

  def _load_one(self, anim_root: Path, name: str) -> "_LoadedAnimation":
    short_name = name.split("/")[-1]
    pkl_path = anim_root / f"{short_name}.pkl"
    info_path = anim_root / f"{short_name}_info.json"
    if not pkl_path.exists() or not info_path.exists():
      raise FileNotFoundError(
        f"Missing HRGym animation files for {name}: {pkl_path}, {info_path}"
      )
    with pkl_path.open("rb") as f:
      raw = pickle.load(f)
    info = json.loads(info_path.read_text())
    root_pos = torch.stack(
      [
        torch.as_tensor(raw["Pelvis_pos_x"], dtype=torch.float32),
        torch.as_tensor(raw["Pelvis_pos_y"], dtype=torch.float32),
        torch.as_tensor(raw["Pelvis_pos_z"], dtype=torch.float32),
      ],
      dim=-1,
    )
    root_quat_xyzw = torch.as_tensor(raw["Pelvis_quat"], dtype=torch.float32)
    joint_pos = torch.stack(
      [torch.as_tensor(raw[name], dtype=torch.float32) for name in HRGYM_HUMAN_JOINT_NAMES],
      dim=-1,
    )
    return _LoadedAnimation(
      name=name,
      info=info,
      root_pos=root_pos,
      root_quat_xyzw=root_quat_xyzw,
      joint_pos=joint_pos,
    )

  def __len__(self) -> int:
    return len(self.animations)


@dataclass(frozen=True)
class _LoadedAnimation:
  name: str
  info: dict
  root_pos: torch.Tensor
  root_quat_xyzw: torch.Tensor
  joint_pos: torch.Tensor

  def to_device(self, device: torch.device) -> HrgymAnimation:
    return HrgymAnimation(
      name=self.name,
      info=self.info,
      root_pos=self.root_pos.to(device),
      root_quat_xyzw=self.root_quat_xyzw.to(device),
      joint_pos=self.joint_pos.to(device),
    )


def xyzw_to_wxyz(quat: torch.Tensor) -> torch.Tensor:
  return torch.stack([quat[..., 3], quat[..., 0], quat[..., 1], quat[..., 2]], dim=-1)
