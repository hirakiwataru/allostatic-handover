"""RSL-RL configs for Mjlab allostatic handover."""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.manipulation.config.yam.rl_cfg import yam_lift_cube_ppo_runner_cfg


def allostatic_handover_yam_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = yam_lift_cube_ppo_runner_cfg()
  cfg.experiment_name = "allostatic_handover_yam"
  cfg.save_interval = 100
  cfg.num_steps_per_env = 32
  cfg.max_iterations = 5_000
  return cfg


def allostatic_handover_full_yam_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = allostatic_handover_yam_ppo_runner_cfg()
  cfg.experiment_name = "allostatic_handover_full_yam"
  # HRGym handover animations need roughly 200 env steps before the human reaches
  # the table, so Full rollouts must be longer than the tabletop lift default.
  cfg.num_steps_per_env = 256
  return cfg


def allostatic_handover_full_task_only_yam_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = allostatic_handover_full_yam_ppo_runner_cfg()
  cfg.experiment_name = "allostatic_handover_full_task_only_yam"
  cfg.actor.distribution_cfg["init_std"] = 1.5
  cfg.algorithm.entropy_coef = 0.01
  cfg.clip_actions = 2.0
  return cfg


def allostatic_handover_full_task_only_speech_yam_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = allostatic_handover_full_task_only_yam_ppo_runner_cfg()
  cfg.experiment_name = "allostatic_handover_full_task_only_speech_yam"
  return cfg


def allostatic_handover_full_speech_penalty_yam_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = allostatic_handover_full_task_only_speech_yam_ppo_runner_cfg()
  cfg.experiment_name = "allostatic_handover_full_speech_penalty_yam"
  return cfg


def allostatic_handover_full_allostatic_belief_yam_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = allostatic_handover_full_task_only_speech_yam_ppo_runner_cfg()
  cfg.experiment_name = "allostatic_handover_full_allostatic_belief_yam"
  return cfg


def allostatic_handover_full_grasped_start_yam_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = allostatic_handover_full_yam_ppo_runner_cfg()
  cfg.experiment_name = "allostatic_handover_full_grasped_start_yam"
  return cfg


def allostatic_handover_full_task_only_grasped_start_yam_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = allostatic_handover_full_yam_ppo_runner_cfg()
  cfg.experiment_name = "allostatic_handover_full_task_only_grasped_start_yam"
  return cfg
