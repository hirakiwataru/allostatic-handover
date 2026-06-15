"""External Mjlab task registration for allostatic handover."""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import (
  allostatic_handover_full_allostatic_belief_yam_env_cfg,
  allostatic_handover_full_dreamerv3_allostatic_yam_env_cfg,
  allostatic_handover_full_grasped_start_yam_env_cfg,
  allostatic_handover_full_speech_penalty_yam_env_cfg,
  allostatic_handover_full_task_only_grasped_start_yam_env_cfg,
  allostatic_handover_full_task_only_speech_yam_env_cfg,
  allostatic_handover_full_task_only_yam_env_cfg,
  allostatic_handover_full_yam_env_cfg,
  allostatic_handover_yam_env_cfg,
)
from .rl_cfg import (
  allostatic_handover_full_allostatic_belief_yam_ppo_runner_cfg,
  allostatic_handover_full_dreamerv3_allostatic_yam_ppo_runner_cfg,
  allostatic_handover_full_grasped_start_yam_ppo_runner_cfg,
  allostatic_handover_full_speech_penalty_yam_ppo_runner_cfg,
  allostatic_handover_full_task_only_grasped_start_yam_ppo_runner_cfg,
  allostatic_handover_full_task_only_speech_yam_ppo_runner_cfg,
  allostatic_handover_full_task_only_yam_ppo_runner_cfg,
  allostatic_handover_full_yam_ppo_runner_cfg,
  allostatic_handover_yam_ppo_runner_cfg,
)


register_mjlab_task(
  task_id="Mjlab-Allostatic-Handover-Yam",
  env_cfg=allostatic_handover_yam_env_cfg(),
  play_env_cfg=allostatic_handover_yam_env_cfg(play=True),
  rl_cfg=allostatic_handover_yam_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-Allostatic-Handover-Full",
  env_cfg=allostatic_handover_full_yam_env_cfg(),
  play_env_cfg=allostatic_handover_full_yam_env_cfg(play=True),
  rl_cfg=allostatic_handover_full_yam_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-Allostatic-Handover-Full-GraspedStart",
  env_cfg=allostatic_handover_full_grasped_start_yam_env_cfg(),
  play_env_cfg=allostatic_handover_full_grasped_start_yam_env_cfg(play=True),
  rl_cfg=allostatic_handover_full_grasped_start_yam_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-Allostatic-Handover-Full-TaskOnly",
  env_cfg=allostatic_handover_full_task_only_yam_env_cfg(),
  play_env_cfg=allostatic_handover_full_task_only_yam_env_cfg(play=True),
  rl_cfg=allostatic_handover_full_task_only_yam_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-Allostatic-Handover-Full-TaskOnlySpeech",
  env_cfg=allostatic_handover_full_task_only_speech_yam_env_cfg(),
  play_env_cfg=allostatic_handover_full_task_only_speech_yam_env_cfg(play=True),
  rl_cfg=allostatic_handover_full_task_only_speech_yam_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-Allostatic-Handover-Full-SpeechPenalty",
  env_cfg=allostatic_handover_full_speech_penalty_yam_env_cfg(),
  play_env_cfg=allostatic_handover_full_speech_penalty_yam_env_cfg(play=True),
  rl_cfg=allostatic_handover_full_speech_penalty_yam_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-Allostatic-Handover-Full-AllostaticBelief",
  env_cfg=allostatic_handover_full_allostatic_belief_yam_env_cfg(),
  play_env_cfg=allostatic_handover_full_allostatic_belief_yam_env_cfg(play=True),
  rl_cfg=allostatic_handover_full_allostatic_belief_yam_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-Allostatic-Handover-Full-DreamerV3Allostatic",
  env_cfg=allostatic_handover_full_dreamerv3_allostatic_yam_env_cfg(),
  play_env_cfg=allostatic_handover_full_dreamerv3_allostatic_yam_env_cfg(play=True),
  rl_cfg=allostatic_handover_full_dreamerv3_allostatic_yam_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-Allostatic-Handover-Full-TaskOnly-GraspedStart",
  env_cfg=allostatic_handover_full_task_only_grasped_start_yam_env_cfg(),
  play_env_cfg=allostatic_handover_full_task_only_grasped_start_yam_env_cfg(play=True),
  rl_cfg=allostatic_handover_full_task_only_grasped_start_yam_ppo_runner_cfg(),
)
