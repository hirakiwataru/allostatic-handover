.PHONY: smoke scripted dashboard mujoco-viewer mjlab-install mjlab-list-envs mjlab-check-display mjlab-check-full-layout mjlab-diagnose-full-task-only mjlab-render-random-animations mjlab-play-yam mjlab-play-full mjlab-play-full-allostatic mjlab-play-full-random-animations mjlab-play-full-task-only mjlab-play-full-task-only-speech mjlab-play-full-task-only-speech-trained mjlab-play-full-task-only-speech-trained-live mjlab-play-full-speech-penalty mjlab-play-full-speech-penalty-trained mjlab-play-full-allostatic-belief-trained mjlab-play-full-task-only-trained mjlab-play-full-task-only-trained-no-reset mjlab-play-full-task-only-grasped-start mjlab-train-yam-smoke mjlab-train-yam-smoke-cpu mjlab-train-yam-wandb-gpu mjlab-train-full-smoke mjlab-train-full-wandb-gpu mjlab-train-full-grasped-start-smoke mjlab-train-full-grasped-start-wandb-gpu tmux-mjlab-train-full-grasped-start-wandb-gpu tail-mjlab-full-grasped-start mjlab-train-full-task-only-smoke mjlab-train-full-task-only-visual-explore mjlab-train-full-task-only-wandb-gpu tmux-mjlab-train-full-task-only-wandb-gpu tail-mjlab-full-task-only mjlab-train-full-task-only-speech-smoke mjlab-train-full-task-only-speech-wandb-gpu tmux-mjlab-train-full-task-only-speech-wandb-gpu tail-mjlab-full-task-only-speech mjlab-train-full-speech-penalty-smoke mjlab-train-full-speech-penalty-wandb-gpu tmux-mjlab-train-full-speech-penalty-wandb-gpu tail-mjlab-full-speech-penalty mjlab-train-three-condition-sequence-wandb-gpu tmux-mjlab-train-three-condition-sequence-wandb-gpu tail-mjlab-three-condition-sequence mjlab-collect-world-model-dataset-smoke mjlab-collect-world-model-dataset-smoke-cpu dreamer-train-world-model-smoke dreamerv3-exact-venv dreamerv3-exact-check-deps dreamerv3-exact-check-jax-cpu dreamerv3-exact-check-jax-gpu dreamer-train-world-model-exact-smoke dreamer-train-world-model-exact-gpu-smoke tmux-dreamer-train-world-model-exact tail-dreamer-world-model-exact mjlab-train-full-allostatic-belief-smoke tmux-dreamer-train-world-model tmux-mjlab-train-full-allostatic-belief-wandb-gpu tail-dreamer-world-model tail-mjlab-full-allostatic-belief mjlab-eval-full-task-only mjlab-eval-full-task-only-speech mjlab-eval-full-speech-penalty mjlab-eval-full-allostatic-belief mjlab-eval-full-grasped-start mjlab-eval-full-task-only-grasped-start copy-hrgym-assets copy-hrgym-full-assets eval-ppo-original-handover-stable-gui ppo-hrgym-smoke eval-ppo-hrgym-smoke ppo-allostatic-safe-ik-smoke eval-ppo-allostatic-safe-ik-smoke ppo-allostatic-compare-safe-ik tmux-ppo-allostatic-compare-safe-ik tail-ppo-allostatic-compare-safe-ik ppo-original-handover-wandb ppo-original-handover-bc-smoke eval-ppo-original-handover-bc-smoke test

MJLAB_DISPLAY ?= :1
MJLAB_XAUTHORITY ?= /run/user/$(shell id -u)/gdm/Xauthority
MJLAB_PLAY ?= /mnt/k_iwamoto/sim_data/Projects/mjlab/.venv/bin/python scripts/run_mjlab_play.py
WM_DATASET ?= /mnt/k_iwamoto/sim_data/Projects/allostatic-handover/outputs/world_model/latest/task_only_speech_dataset.npz
WM_OUTPUT_DIR ?= /mnt/k_iwamoto/sim_data/Projects/allostatic-handover/outputs/world_model/latest
WM_MODEL ?= /mnt/k_iwamoto/sim_data/Projects/allostatic-handover/outputs/world_model/latest/belief_distill.pt
DREAMERV3_PY ?= /mnt/k_iwamoto/sim_data/Projects/allostatic-handover/.dreamerv3-venv/bin/python
THREE_CONDITION_SESSION ?= allostatic_three_condition_ppo_sequence
THREE_CONDITION_RUN_ID := $(if $(THREE_CONDITION_RUN_ID),$(THREE_CONDITION_RUN_ID),$(shell date +%Y%m%d_%H%M%S))
GPU_ID ?= 0
NUM_ENVS ?= 64
MAX_ITER ?= 500
WM_STEPS ?= 4096
WM_UPDATES ?= 5000
WM_NUM_ENVS ?= $(NUM_ENVS)
WM_BATCH_SIZE ?= 64
WM_SEQ_LEN ?= 64

smoke:
	python -m allostatic_handover.experiments.run_scripted_rollouts --backend mock --policy excessive_speech --episodes 2 --horizon 80 --output-dir outputs/smoke_excessive

scripted:
	python -m allostatic_handover.experiments.eval_degenerate_policy --backend mock --episodes 4 --horizon 120

dashboard:
	python -m allostatic_handover.dashboard.app --log-dir outputs --port 7860

mujoco-viewer:
	MUJOCO_GL=glfw python -m allostatic_handover.experiments.run_scripted_rollouts --backend hrgym --policy minimal_speech --reward-variant task_only --episodes 1 --horizon 1000 --render --print-step-info --output-dir outputs/hrgym_mujoco_viewer

mjlab-install:
	UV_CACHE_DIR=/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/.uvcache uv pip install --python /mnt/k_iwamoto/sim_data/Projects/mjlab/.venv/bin/python --no-deps --no-build-isolation -e .

mjlab-list-envs:
	cd /mnt/k_iwamoto/sim_data/Projects/mjlab && UV_CACHE_DIR=/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/.uvcache MPLCONFIGDIR=/mnt/k_iwamoto/sim_data/tmp/matplotlib uv run list-envs | grep Allostatic

mjlab-check-display:
	DISPLAY=$(MJLAB_DISPLAY) XAUTHORITY=$(MJLAB_XAUTHORITY) xdpyinfo

mjlab-check-full-layout:
	XDG_CACHE_HOME=/mnt/k_iwamoto/sim_data/tmp/xdg_cache MPLCONFIGDIR=/mnt/k_iwamoto/sim_data/tmp/matplotlib MUJOCO_GL=egl python3 scripts/compare_hrgym_mjlab_full_layout.py

mjlab-diagnose-full-task-only:
	XDG_CACHE_HOME=/mnt/k_iwamoto/sim_data/tmp/xdg_cache MPLCONFIGDIR=/mnt/k_iwamoto/sim_data/tmp/matplotlib MUJOCO_GL=egl /mnt/k_iwamoto/sim_data/Projects/mjlab/.venv/bin/python scripts/diagnose_mjlab_full_layout.py Mjlab-Allostatic-Handover-Full-TaskOnly --render-path outputs/visual_checks/mjlab_full_task_only/reset_diagnostic.png

mjlab-render-random-animations:
	XDG_CACHE_HOME=/mnt/k_iwamoto/sim_data/tmp/xdg_cache MPLCONFIGDIR=/mnt/k_iwamoto/sim_data/tmp/matplotlib MUJOCO_GL=egl /mnt/k_iwamoto/sim_data/Projects/mjlab/.venv/bin/python scripts/render_mjlab_random_animations.py --samples 6 --seed 17

mjlab-play-yam:
	$(MJLAB_PLAY) Mjlab-Allostatic-Handover-Yam --agent zero --viewer native --display $(MJLAB_DISPLAY) --xauthority $(MJLAB_XAUTHORITY)

mjlab-play-full:
	$(MJLAB_PLAY) Mjlab-Allostatic-Handover-Full-TaskOnly --agent zero --viewer native --display $(MJLAB_DISPLAY) --xauthority $(MJLAB_XAUTHORITY)

mjlab-play-full-allostatic:
	$(MJLAB_PLAY) Mjlab-Allostatic-Handover-Full --agent zero --viewer native --display $(MJLAB_DISPLAY) --xauthority $(MJLAB_XAUTHORITY)

mjlab-play-full-random-animations: mjlab-play-full-allostatic

mjlab-play-full-task-only:
	$(MJLAB_PLAY) Mjlab-Allostatic-Handover-Full-TaskOnly --agent zero --viewer native --display $(MJLAB_DISPLAY) --xauthority $(MJLAB_XAUTHORITY)

mjlab-play-full-task-only-speech:
	$(MJLAB_PLAY) Mjlab-Allostatic-Handover-Full-TaskOnlySpeech --agent zero --viewer native --display $(MJLAB_DISPLAY) --xauthority $(MJLAB_XAUTHORITY)

mjlab-play-full-task-only-speech-trained:
	$(MJLAB_PLAY) Mjlab-Allostatic-Handover-Full-TaskOnlySpeech --agent trained --checkpoint-file "$(CKPT)" --viewer native --display $(MJLAB_DISPLAY) --xauthority $(MJLAB_XAUTHORITY)

mjlab-play-full-task-only-speech-trained-live:
	$(MJLAB_PLAY) Mjlab-Allostatic-Handover-Full-TaskOnlySpeech --agent trained --checkpoint-file "$(CKPT)" --viewer native --display $(MJLAB_DISPLAY) --xauthority $(MJLAB_XAUTHORITY) --live-dashboard --live-log-interval 5 --dashboard-port 7860

mjlab-play-full-speech-penalty:
	$(MJLAB_PLAY) Mjlab-Allostatic-Handover-Full-SpeechPenalty --agent zero --viewer native --display $(MJLAB_DISPLAY) --xauthority $(MJLAB_XAUTHORITY)

mjlab-play-full-speech-penalty-trained:
	$(MJLAB_PLAY) Mjlab-Allostatic-Handover-Full-SpeechPenalty --agent trained --checkpoint-file "$(CKPT)" --viewer native --display $(MJLAB_DISPLAY) --xauthority $(MJLAB_XAUTHORITY)

mjlab-play-full-allostatic-belief-trained:
	ALLOSTATIC_WM_BELIEF_MODEL=$(WM_MODEL) $(MJLAB_PLAY) Mjlab-Allostatic-Handover-Full-AllostaticBelief --agent trained --checkpoint-file "$(CKPT)" --viewer native --display $(MJLAB_DISPLAY) --xauthority $(MJLAB_XAUTHORITY)

mjlab-play-full-task-only-trained:
	$(MJLAB_PLAY) Mjlab-Allostatic-Handover-Full-TaskOnly --agent trained --checkpoint-file "$(CKPT)" --viewer native --display $(MJLAB_DISPLAY) --xauthority $(MJLAB_XAUTHORITY)

mjlab-play-full-task-only-trained-no-reset:
	$(MJLAB_PLAY) Mjlab-Allostatic-Handover-Full-TaskOnly --agent trained --checkpoint-file "$(CKPT)" --viewer native --display $(MJLAB_DISPLAY) --xauthority $(MJLAB_XAUTHORITY) --no-terminations True

mjlab-play-full-task-only-grasped-start:
	$(MJLAB_PLAY) Mjlab-Allostatic-Handover-Full-TaskOnly-GraspedStart --checkpoint-file "$(CKPT)" --agent trained --viewer native --display $(MJLAB_DISPLAY) --xauthority $(MJLAB_XAUTHORITY)

mjlab-train-yam-smoke:
	cd /mnt/k_iwamoto/sim_data/Projects/mjlab && UV_CACHE_DIR=/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/.uvcache XDG_CACHE_HOME=/mnt/k_iwamoto/sim_data/tmp/xdg_cache MPLCONFIGDIR=/mnt/k_iwamoto/sim_data/tmp/matplotlib CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run train Mjlab-Allostatic-Handover-Yam --gpu-ids '[0]' --agent.logger tensorboard --env.scene.num-envs 64 --agent.max-iterations 2

mjlab-train-yam-smoke-cpu:
	cd /mnt/k_iwamoto/sim_data/Projects/mjlab && UV_CACHE_DIR=/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/.uvcache XDG_CACHE_HOME=/mnt/k_iwamoto/sim_data/tmp/xdg_cache MPLCONFIGDIR=/mnt/k_iwamoto/sim_data/tmp/matplotlib MUJOCO_GL=egl uv run train Mjlab-Allostatic-Handover-Yam --gpu-ids None --agent.logger tensorboard --env.scene.num-envs 64 --agent.max-iterations 2

mjlab-train-yam-wandb-gpu:
	cd /mnt/k_iwamoto/sim_data/Projects/mjlab && UV_CACHE_DIR=/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/.uvcache XDG_CACHE_HOME=/mnt/k_iwamoto/sim_data/tmp/xdg_cache MPLCONFIGDIR=/mnt/k_iwamoto/sim_data/tmp/matplotlib CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run train Mjlab-Allostatic-Handover-Yam --gpu-ids '[0]' --agent.logger wandb --agent.wandb-project allostatic-handover-mjlab --agent.experiment-name allostatic_handover_yam --env.scene.num-envs 1024 --agent.max-iterations 5000

mjlab-train-full-smoke:
	cd /mnt/k_iwamoto/sim_data/Projects/mjlab && UV_CACHE_DIR=/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/.uvcache XDG_CACHE_HOME=/mnt/k_iwamoto/sim_data/tmp/xdg_cache MPLCONFIGDIR=/mnt/k_iwamoto/sim_data/tmp/matplotlib CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run train Mjlab-Allostatic-Handover-Full --gpu-ids '[0]' --agent.logger tensorboard --env.scene.num-envs 64 --agent.max-iterations 2

mjlab-train-full-wandb-gpu:
	cd /mnt/k_iwamoto/sim_data/Projects/mjlab && UV_CACHE_DIR=/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/.uvcache XDG_CACHE_HOME=/mnt/k_iwamoto/sim_data/tmp/xdg_cache MPLCONFIGDIR=/mnt/k_iwamoto/sim_data/tmp/matplotlib CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run train Mjlab-Allostatic-Handover-Full --gpu-ids '[0]' --agent.logger wandb --agent.wandb-project allostatic-handover-mjlab --agent.experiment-name allostatic_handover_full_yam --env.scene.num-envs 1024 --agent.max-iterations 5000

mjlab-train-full-grasped-start-smoke:
	cd /mnt/k_iwamoto/sim_data/Projects/mjlab && UV_CACHE_DIR=/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/.uvcache XDG_CACHE_HOME=/mnt/k_iwamoto/sim_data/tmp/xdg_cache MPLCONFIGDIR=/mnt/k_iwamoto/sim_data/tmp/matplotlib CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run train Mjlab-Allostatic-Handover-Full-GraspedStart --gpu-ids '[0]' --agent.logger tensorboard --agent.experiment-name allostatic_handover_full_grasped_start_yam --agent.run-name smoke --env.scene.num-envs 32 --agent.max-iterations 60

mjlab-train-full-grasped-start-wandb-gpu:
	cd /mnt/k_iwamoto/sim_data/Projects/mjlab && UV_CACHE_DIR=/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/.uvcache XDG_CACHE_HOME=/mnt/k_iwamoto/sim_data/tmp/xdg_cache MPLCONFIGDIR=/mnt/k_iwamoto/sim_data/tmp/matplotlib CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run train Mjlab-Allostatic-Handover-Full-GraspedStart --gpu-ids '[0]' --agent.logger wandb --agent.wandb-project allostatic-handover-mjlab --agent.experiment-name allostatic_handover_full_grasped_start_yam --agent.run-name ppo_full_allostatic_grasped_start_release_intent_64env --env.scene.num-envs 64 --agent.max-iterations 1000

tmux-mjlab-train-full-grasped-start-wandb-gpu:
	mkdir -p outputs/mjlab_full_allostatic_grasped_start_release_intent_64env && ./.conda/bin/tmux new-session -d -s mjlab_full_allostatic_grasped_start_ppo "cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover && make mjlab-train-full-grasped-start-wandb-gpu > outputs/mjlab_full_allostatic_grasped_start_release_intent_64env/run.log 2>&1"

tail-mjlab-full-grasped-start:
	tail -f outputs/mjlab_full_allostatic_grasped_start_release_intent_64env/run.log

mjlab-train-full-task-only-smoke:
	cd /mnt/k_iwamoto/sim_data/Projects/mjlab && UV_CACHE_DIR=/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/.uvcache XDG_CACHE_HOME=/mnt/k_iwamoto/sim_data/tmp/xdg_cache MPLCONFIGDIR=/mnt/k_iwamoto/sim_data/tmp/matplotlib CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run train Mjlab-Allostatic-Handover-Full-TaskOnly --gpu-ids '[0]' --agent.logger tensorboard --env.scene.num-envs 64 --agent.max-iterations 2 --agent.run-name smoke_fixed_hammer_liftgate

mjlab-train-full-task-only-visual-explore:
	cd /mnt/k_iwamoto/sim_data/Projects/mjlab && DISPLAY=$(MJLAB_DISPLAY) XAUTHORITY=$(MJLAB_XAUTHORITY) UV_CACHE_DIR=/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/.uvcache XDG_CACHE_HOME=/mnt/k_iwamoto/sim_data/tmp/xdg_cache MPLCONFIGDIR=/mnt/k_iwamoto/sim_data/tmp/matplotlib CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=glfw uv run train Mjlab-Allostatic-Handover-Full-TaskOnly --gpu-ids '[0]' --env.scene.num-envs 1 --env.actions.arm-ik.delta-pos-scale 0.12 --env.actions.arm-ik.max-dq 0.25 --agent.actor.distribution-cfg.init-std 2.0 --agent.clip-actions 2.0 --agent.logger tensorboard --agent.experiment-name allostatic_handover_full_task_only_yam --agent.run-name visual_explore_fixed_hammer_liftgate_1env_10iter --agent.max-iterations 10 --video True --video-length 1000 --video-interval 1

mjlab-train-full-task-only-wandb-gpu:
	cd /mnt/k_iwamoto/sim_data/Projects/mjlab && UV_CACHE_DIR=/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/.uvcache XDG_CACHE_HOME=/mnt/k_iwamoto/sim_data/tmp/xdg_cache MPLCONFIGDIR=/mnt/k_iwamoto/sim_data/tmp/matplotlib CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run train Mjlab-Allostatic-Handover-Full-TaskOnly --gpu-ids '[0]' --agent.logger wandb --agent.wandb-project allostatic-handover-mjlab --agent.experiment-name allostatic_handover_full_task_only_yam --agent.run-name ppo_task_only_fixed_animation_object_grasp_shaping_v2_64env_500iter --env.scene.num-envs 64 --agent.max-iterations 500

tmux-mjlab-train-full-task-only-wandb-gpu:
	mkdir -p outputs/mjlab_full_task_only_fixed_animation_object_grasp_shaping_v2_64env_500iter && ./.conda/bin/tmux new-session -d -s mjlab_full_task_only_fixed_animation_object_ppo "cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover && make mjlab-train-full-task-only-wandb-gpu > outputs/mjlab_full_task_only_fixed_animation_object_grasp_shaping_v2_64env_500iter/run.log 2>&1"

tail-mjlab-full-task-only:
	tail -f outputs/mjlab_full_task_only_fixed_animation_object_grasp_shaping_v2_64env_500iter/run.log

mjlab-train-full-task-only-speech-smoke:
	cd /mnt/k_iwamoto/sim_data/Projects/mjlab && UV_CACHE_DIR=/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/.uvcache XDG_CACHE_HOME=/mnt/k_iwamoto/sim_data/tmp/xdg_cache MPLCONFIGDIR=/mnt/k_iwamoto/sim_data/tmp/matplotlib CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run train Mjlab-Allostatic-Handover-Full-TaskOnlySpeech --gpu-ids '[0]' --agent.logger tensorboard --env.scene.num-envs 32 --agent.max-iterations 2 --agent.run-name smoke_task_only_speech

mjlab-train-full-task-only-speech-wandb-gpu:
	cd /mnt/k_iwamoto/sim_data/Projects/mjlab && UV_CACHE_DIR=/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/.uvcache XDG_CACHE_HOME=/mnt/k_iwamoto/sim_data/tmp/xdg_cache MPLCONFIGDIR=/mnt/k_iwamoto/sim_data/tmp/matplotlib CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run train Mjlab-Allostatic-Handover-Full-TaskOnlySpeech --gpu-ids '[0]' --agent.logger wandb --agent.wandb-project allostatic-handover-mjlab --agent.experiment-name allostatic_handover_full_task_only_speech_yam --agent.run-name ppo_task_only_speech_fixed_animation_object_64env_500iter --env.scene.num-envs 64 --agent.max-iterations 500

tmux-mjlab-train-full-task-only-speech-wandb-gpu:
	mkdir -p outputs/mjlab_full_task_only_speech_fixed_animation_object_64env_500iter && ./.conda/bin/tmux new-session -d -s mjlab_full_task_only_speech_ppo "cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover && make mjlab-train-full-task-only-speech-wandb-gpu > outputs/mjlab_full_task_only_speech_fixed_animation_object_64env_500iter/run.log 2>&1"

tail-mjlab-full-task-only-speech:
	tail -f outputs/mjlab_full_task_only_speech_fixed_animation_object_64env_500iter/run.log

mjlab-train-full-speech-penalty-smoke:
	cd /mnt/k_iwamoto/sim_data/Projects/mjlab && UV_CACHE_DIR=/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/.uvcache XDG_CACHE_HOME=/mnt/k_iwamoto/sim_data/tmp/xdg_cache MPLCONFIGDIR=/mnt/k_iwamoto/sim_data/tmp/matplotlib CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run train Mjlab-Allostatic-Handover-Full-SpeechPenalty --gpu-ids '[0]' --agent.logger tensorboard --env.scene.num-envs 32 --agent.max-iterations 2 --agent.run-name smoke_speech_penalty

mjlab-train-full-speech-penalty-wandb-gpu:
	cd /mnt/k_iwamoto/sim_data/Projects/mjlab && UV_CACHE_DIR=/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/.uvcache XDG_CACHE_HOME=/mnt/k_iwamoto/sim_data/tmp/xdg_cache MPLCONFIGDIR=/mnt/k_iwamoto/sim_data/tmp/matplotlib CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run train Mjlab-Allostatic-Handover-Full-SpeechPenalty --gpu-ids '[0]' --agent.logger wandb --agent.wandb-project allostatic-handover-mjlab --agent.experiment-name allostatic_handover_full_speech_penalty_yam --agent.run-name ppo_speech_penalty_fixed_animation_object_64env_500iter --env.scene.num-envs 64 --agent.max-iterations 500

tmux-mjlab-train-full-speech-penalty-wandb-gpu:
	mkdir -p outputs/mjlab_full_speech_penalty_fixed_animation_object_64env_500iter && ./.conda/bin/tmux new-session -d -s mjlab_full_speech_penalty_ppo "cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover && make mjlab-train-full-speech-penalty-wandb-gpu > outputs/mjlab_full_speech_penalty_fixed_animation_object_64env_500iter/run.log 2>&1"

tail-mjlab-full-speech-penalty:
	tail -f outputs/mjlab_full_speech_penalty_fixed_animation_object_64env_500iter/run.log

mjlab-train-three-condition-sequence-wandb-gpu:
	RUN_ID=$(THREE_CONDITION_RUN_ID) GPU_ID=$(GPU_ID) NUM_ENVS=$(NUM_ENVS) MAX_ITER=$(MAX_ITER) WM_NUM_ENVS=$(WM_NUM_ENVS) WM_STEPS=$(WM_STEPS) WM_UPDATES=$(WM_UPDATES) WM_BATCH_SIZE=$(WM_BATCH_SIZE) WM_SEQ_LEN=$(WM_SEQ_LEN) scripts/run_mjlab_three_condition_ppo_sequence.sh

tmux-mjlab-train-three-condition-sequence-wandb-gpu:
	mkdir -p outputs/mjlab_three_condition_compare/$(THREE_CONDITION_RUN_ID) && ./.conda/bin/tmux new-session -d -s $(THREE_CONDITION_SESSION) "cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover && RUN_ID=$(THREE_CONDITION_RUN_ID) GPU_ID=$(GPU_ID) NUM_ENVS=$(NUM_ENVS) MAX_ITER=$(MAX_ITER) WM_NUM_ENVS=$(WM_NUM_ENVS) WM_STEPS=$(WM_STEPS) WM_UPDATES=$(WM_UPDATES) WM_BATCH_SIZE=$(WM_BATCH_SIZE) WM_SEQ_LEN=$(WM_SEQ_LEN) scripts/run_mjlab_three_condition_ppo_sequence.sh > outputs/mjlab_three_condition_compare/$(THREE_CONDITION_RUN_ID)/tmux.log 2>&1"
	@echo "Started tmux session $(THREE_CONDITION_SESSION)"
	@echo "Run log: outputs/mjlab_three_condition_compare/$(THREE_CONDITION_RUN_ID)/run.log"
	@echo "Attach: ./.conda/bin/tmux attach -t $(THREE_CONDITION_SESSION)"

tail-mjlab-three-condition-sequence:
	tail -f $$(ls -dt outputs/mjlab_three_condition_compare/*/run.log | head -n 1)

mjlab-collect-world-model-dataset-smoke: mjlab-install
	cd /mnt/k_iwamoto/sim_data/Projects/mjlab && UV_CACHE_DIR=/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/.uvcache XDG_CACHE_HOME=/mnt/k_iwamoto/sim_data/tmp/xdg_cache MPLCONFIGDIR=/mnt/k_iwamoto/sim_data/tmp/matplotlib CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run python /mnt/k_iwamoto/sim_data/Projects/allostatic-handover/scripts/collect_mjlab_world_model_dataset.py --output $(WM_DATASET) --num-envs 8 --steps 256 --seed 101 --device cuda:0 --policy mixed

mjlab-collect-world-model-dataset-smoke-cpu: mjlab-install
	cd /mnt/k_iwamoto/sim_data/Projects/mjlab && UV_CACHE_DIR=/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/.uvcache XDG_CACHE_HOME=/mnt/k_iwamoto/sim_data/tmp/xdg_cache MPLCONFIGDIR=/mnt/k_iwamoto/sim_data/tmp/matplotlib CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl uv run python /mnt/k_iwamoto/sim_data/Projects/allostatic-handover/scripts/collect_mjlab_world_model_dataset.py --output $(WM_DATASET) --num-envs 2 --steps 128 --seed 101 --device cpu --policy mixed

dreamer-train-world-model-smoke: mjlab-collect-world-model-dataset-smoke
	cd /mnt/k_iwamoto/sim_data/Projects/mjlab && PYTHONPATH=/mnt/k_iwamoto/sim_data/Projects/dreamerv3:/mnt/k_iwamoto/sim_data/Projects/allostatic-handover UV_CACHE_DIR=/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/.uvcache XDG_CACHE_HOME=/mnt/k_iwamoto/sim_data/tmp/xdg_cache MPLCONFIGDIR=/mnt/k_iwamoto/sim_data/tmp/matplotlib CUDA_VISIBLE_DEVICES=0 uv run python /mnt/k_iwamoto/sim_data/Projects/allostatic-handover/scripts/train_dreamer_world_model.py --dataset $(WM_DATASET) --output-dir $(WM_OUTPUT_DIR) --updates 100 --batch-size 16 --seq-len 32 --device cuda:0 --dreamerv3-path /mnt/k_iwamoto/sim_data/Projects/dreamerv3

dreamerv3-exact-venv:
	UV_CACHE_DIR=/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/.uvcache uv venv --python python3 /mnt/k_iwamoto/sim_data/Projects/allostatic-handover/.dreamerv3-venv
	UV_CACHE_DIR=/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/.uvcache uv pip install --python $(DREAMERV3_PY) -r /mnt/k_iwamoto/sim_data/Projects/dreamerv3/requirements.txt
	UV_CACHE_DIR=/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/.uvcache uv pip install --python $(DREAMERV3_PY) --no-deps -e /mnt/k_iwamoto/sim_data/Projects/dreamerv3
	UV_CACHE_DIR=/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/.uvcache uv pip install --python $(DREAMERV3_PY) --no-deps -e /mnt/k_iwamoto/sim_data/Projects/allostatic-handover

dreamerv3-exact-check-deps:
	PYTHONPATH=/mnt/k_iwamoto/sim_data/Projects/dreamerv3:/mnt/k_iwamoto/sim_data/Projects/allostatic-handover $(DREAMERV3_PY) scripts/check_dreamerv3_exact_dependencies.py --fail

dreamerv3-exact-check-jax-cpu:
	PYTHONPATH=/mnt/k_iwamoto/sim_data/Projects/dreamerv3:/mnt/k_iwamoto/sim_data/Projects/allostatic-handover CUDA_VISIBLE_DEVICES= XLA_PYTHON_CLIENT_PREALLOCATE=false $(DREAMERV3_PY) scripts/check_dreamerv3_jax_runtime.py --platform cpu --fail

dreamerv3-exact-check-jax-gpu:
	PYTHONPATH=/mnt/k_iwamoto/sim_data/Projects/dreamerv3:/mnt/k_iwamoto/sim_data/Projects/allostatic-handover CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false $(DREAMERV3_PY) scripts/check_dreamerv3_jax_runtime.py --platform cuda --fail

dreamer-train-world-model-exact-smoke: mjlab-collect-world-model-dataset-smoke-cpu dreamerv3-exact-check-deps
	PYTHONPATH=/mnt/k_iwamoto/sim_data/Projects/dreamerv3:/mnt/k_iwamoto/sim_data/Projects/allostatic-handover CUDA_VISIBLE_DEVICES= XLA_PYTHON_CLIENT_PREALLOCATE=false $(DREAMERV3_PY) scripts/train_dreamerv3_exact_world_model.py --dataset $(WM_DATASET) --output-dir /mnt/k_iwamoto/sim_data/Projects/allostatic-handover/outputs/world_model/dreamerv3_exact_smoke --updates 2 --batch-size 4 --batch-length 8 --configs debug --jax-platform cpu --dreamerv3-path /mnt/k_iwamoto/sim_data/Projects/dreamerv3

dreamer-train-world-model-exact-gpu-smoke: mjlab-collect-world-model-dataset-smoke dreamerv3-exact-check-deps dreamerv3-exact-check-jax-gpu
	PYTHONPATH=/mnt/k_iwamoto/sim_data/Projects/dreamerv3:/mnt/k_iwamoto/sim_data/Projects/allostatic-handover CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false $(DREAMERV3_PY) scripts/train_dreamerv3_exact_world_model.py --dataset $(WM_DATASET) --output-dir /mnt/k_iwamoto/sim_data/Projects/allostatic-handover/outputs/world_model/dreamerv3_exact_gpu_smoke --updates 2 --batch-size 4 --batch-length 8 --configs debug --jax-platform cuda --dreamerv3-path /mnt/k_iwamoto/sim_data/Projects/dreamerv3

tmux-dreamer-train-world-model-exact:
	mkdir -p outputs/world_model/dreamerv3_exact_full && ./.conda/bin/tmux new-session -d -s dreamer_exact_allostatic_world_model "cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover && PYTHONPATH=/mnt/k_iwamoto/sim_data/Projects/dreamerv3:/mnt/k_iwamoto/sim_data/Projects/allostatic-handover CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false $(DREAMERV3_PY) scripts/train_dreamerv3_exact_world_model.py --dataset /mnt/k_iwamoto/sim_data/Projects/allostatic-handover/outputs/world_model/full/task_only_speech_dataset.npz --output-dir /mnt/k_iwamoto/sim_data/Projects/allostatic-handover/outputs/world_model/dreamerv3_exact_full --updates 5000 --batch-size 16 --batch-length 64 --configs size1m --jax-platform cuda --dreamerv3-path /mnt/k_iwamoto/sim_data/Projects/dreamerv3 --wandb-mode online --wandb-project allostatic-handover-mjlab --wandb-run-name dreamerv3_exact_world_model_task_only_speech_seed101 > /mnt/k_iwamoto/sim_data/Projects/allostatic-handover/outputs/world_model/dreamerv3_exact_full/run.log 2>&1"

tail-dreamer-world-model-exact:
	tail -f outputs/world_model/dreamerv3_exact_full/run.log

mjlab-train-full-allostatic-belief-smoke: dreamer-train-world-model-smoke
	cd /mnt/k_iwamoto/sim_data/Projects/mjlab && ALLOSTATIC_WM_BELIEF_MODEL=$(WM_MODEL) UV_CACHE_DIR=/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/.uvcache XDG_CACHE_HOME=/mnt/k_iwamoto/sim_data/tmp/xdg_cache MPLCONFIGDIR=/mnt/k_iwamoto/sim_data/tmp/matplotlib CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run train Mjlab-Allostatic-Handover-Full-AllostaticBelief --gpu-ids '[0]' --agent.logger tensorboard --env.scene.num-envs 32 --agent.max-iterations 2 --agent.run-name smoke_allostatic_belief

tmux-dreamer-train-world-model:
	mkdir -p outputs/world_model/full && ./.conda/bin/tmux new-session -d -s dreamer_allostatic_world_model "cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover && make mjlab-install && cd /mnt/k_iwamoto/sim_data/Projects/mjlab && UV_CACHE_DIR=/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/.uvcache XDG_CACHE_HOME=/mnt/k_iwamoto/sim_data/tmp/xdg_cache MPLCONFIGDIR=/mnt/k_iwamoto/sim_data/tmp/matplotlib CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run python /mnt/k_iwamoto/sim_data/Projects/allostatic-handover/scripts/collect_mjlab_world_model_dataset.py --output /mnt/k_iwamoto/sim_data/Projects/allostatic-handover/outputs/world_model/full/task_only_speech_dataset.npz --num-envs 64 --steps 4096 --seed 101 --device cuda:0 --policy mixed && PYTHONPATH=/mnt/k_iwamoto/sim_data/Projects/dreamerv3:/mnt/k_iwamoto/sim_data/Projects/allostatic-handover UV_CACHE_DIR=/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/.uvcache XDG_CACHE_HOME=/mnt/k_iwamoto/sim_data/tmp/xdg_cache MPLCONFIGDIR=/mnt/k_iwamoto/sim_data/tmp/matplotlib CUDA_VISIBLE_DEVICES=0 uv run python /mnt/k_iwamoto/sim_data/Projects/allostatic-handover/scripts/train_dreamer_world_model.py --dataset /mnt/k_iwamoto/sim_data/Projects/allostatic-handover/outputs/world_model/full/task_only_speech_dataset.npz --output-dir /mnt/k_iwamoto/sim_data/Projects/allostatic-handover/outputs/world_model/full --updates 5000 --batch-size 64 --seq-len 64 --device cuda:0 --dreamerv3-path /mnt/k_iwamoto/sim_data/Projects/dreamerv3 --wandb-mode online --wandb-project allostatic-handover-mjlab --wandb-run-name dreamer_world_model_task_only_speech_mixed_seed101 > /mnt/k_iwamoto/sim_data/Projects/allostatic-handover/outputs/world_model/full/run.log 2>&1"

tmux-mjlab-train-full-allostatic-belief-wandb-gpu:
	mkdir -p outputs/mjlab_full_allostatic_belief_fixed_animation_object_64env_500iter && ./.conda/bin/tmux new-session -d -s mjlab_full_allostatic_belief_ppo "cd /mnt/k_iwamoto/sim_data/Projects/mjlab && ALLOSTATIC_WM_BELIEF_MODEL=$(WM_MODEL) UV_CACHE_DIR=/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/.uvcache XDG_CACHE_HOME=/mnt/k_iwamoto/sim_data/tmp/xdg_cache MPLCONFIGDIR=/mnt/k_iwamoto/sim_data/tmp/matplotlib CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run train Mjlab-Allostatic-Handover-Full-AllostaticBelief --gpu-ids '[0]' --agent.logger wandb --agent.wandb-project allostatic-handover-mjlab --agent.experiment-name allostatic_handover_full_allostatic_belief_yam --agent.run-name ppo_allostatic_belief_fixed_animation_object_64env_500iter --env.scene.num-envs 64 --agent.max-iterations 500 > /mnt/k_iwamoto/sim_data/Projects/allostatic-handover/outputs/mjlab_full_allostatic_belief_fixed_animation_object_64env_500iter/run.log 2>&1"

tail-dreamer-world-model:
	tail -f outputs/world_model/full/run.log

tail-mjlab-full-allostatic-belief:
	tail -f outputs/mjlab_full_allostatic_belief_fixed_animation_object_64env_500iter/run.log

mjlab-eval-full-task-only:
	cd /mnt/k_iwamoto/sim_data/Projects/mjlab && UV_CACHE_DIR=/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/.uvcache XDG_CACHE_HOME=/mnt/k_iwamoto/sim_data/tmp/xdg_cache MPLCONFIGDIR=/mnt/k_iwamoto/sim_data/tmp/matplotlib CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run python /mnt/k_iwamoto/sim_data/Projects/allostatic-handover/scripts/evaluate_mjlab_policy.py Mjlab-Allostatic-Handover-Full-TaskOnly --checkpoint-file $(CKPT) --episodes 64 --num-envs 64 --device cuda:0 --output /mnt/k_iwamoto/sim_data/Projects/allostatic-handover/outputs/mjlab_full_task_only_fixed_hammer_liftgate/eval.json

mjlab-eval-full-task-only-speech:
	cd /mnt/k_iwamoto/sim_data/Projects/mjlab && UV_CACHE_DIR=/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/.uvcache XDG_CACHE_HOME=/mnt/k_iwamoto/sim_data/tmp/xdg_cache MPLCONFIGDIR=/mnt/k_iwamoto/sim_data/tmp/matplotlib CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run python /mnt/k_iwamoto/sim_data/Projects/allostatic-handover/scripts/evaluate_mjlab_policy.py Mjlab-Allostatic-Handover-Full-TaskOnlySpeech --checkpoint-file $(CKPT) --episodes 64 --num-envs 64 --device cuda:0 --output /mnt/k_iwamoto/sim_data/Projects/allostatic-handover/outputs/mjlab_full_task_only_speech/eval.json

mjlab-eval-full-speech-penalty:
	cd /mnt/k_iwamoto/sim_data/Projects/mjlab && UV_CACHE_DIR=/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/.uvcache XDG_CACHE_HOME=/mnt/k_iwamoto/sim_data/tmp/xdg_cache MPLCONFIGDIR=/mnt/k_iwamoto/sim_data/tmp/matplotlib CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run python /mnt/k_iwamoto/sim_data/Projects/allostatic-handover/scripts/evaluate_mjlab_policy.py Mjlab-Allostatic-Handover-Full-SpeechPenalty --checkpoint-file $(CKPT) --episodes 64 --num-envs 64 --device cuda:0 --output /mnt/k_iwamoto/sim_data/Projects/allostatic-handover/outputs/mjlab_full_speech_penalty/eval.json

mjlab-eval-full-allostatic-belief:
	cd /mnt/k_iwamoto/sim_data/Projects/mjlab && ALLOSTATIC_WM_BELIEF_MODEL=$(WM_MODEL) UV_CACHE_DIR=/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/.uvcache XDG_CACHE_HOME=/mnt/k_iwamoto/sim_data/tmp/xdg_cache MPLCONFIGDIR=/mnt/k_iwamoto/sim_data/tmp/matplotlib CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run python /mnt/k_iwamoto/sim_data/Projects/allostatic-handover/scripts/evaluate_mjlab_policy.py Mjlab-Allostatic-Handover-Full-AllostaticBelief --checkpoint-file $(CKPT) --episodes 64 --num-envs 64 --device cuda:0 --output /mnt/k_iwamoto/sim_data/Projects/allostatic-handover/outputs/mjlab_full_allostatic_belief/eval.json

mjlab-eval-full-grasped-start:
	cd /mnt/k_iwamoto/sim_data/Projects/mjlab && UV_CACHE_DIR=/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/.uvcache XDG_CACHE_HOME=/mnt/k_iwamoto/sim_data/tmp/xdg_cache MPLCONFIGDIR=/mnt/k_iwamoto/sim_data/tmp/matplotlib CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run python /mnt/k_iwamoto/sim_data/Projects/allostatic-handover/scripts/evaluate_mjlab_policy.py Mjlab-Allostatic-Handover-Full-GraspedStart --checkpoint-file $(CKPT) --episodes 64 --num-envs 64 --device cuda:0 --output /mnt/k_iwamoto/sim_data/Projects/allostatic-handover/outputs/mjlab_full_allostatic_grasped_start/eval.json

mjlab-eval-full-task-only-grasped-start:
	cd /mnt/k_iwamoto/sim_data/Projects/mjlab && UV_CACHE_DIR=/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/.uvcache XDG_CACHE_HOME=/mnt/k_iwamoto/sim_data/tmp/xdg_cache MPLCONFIGDIR=/mnt/k_iwamoto/sim_data/tmp/matplotlib CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run python /mnt/k_iwamoto/sim_data/Projects/allostatic-handover/scripts/evaluate_mjlab_policy.py Mjlab-Allostatic-Handover-Full-TaskOnly-GraspedStart --checkpoint-file $(CKPT) --episodes 64 --num-envs 64 --device cuda:0 --output /mnt/k_iwamoto/sim_data/Projects/allostatic-handover/outputs/mjlab_full_task_only_grasped_start/eval.json

copy-hrgym-assets:
	python3 scripts/copy_hrgym_assets.py

copy-hrgym-full-assets:
	python3 scripts/copy_hrgym_assets.py --include-full-handover-assets

eval-ppo-original-handover-stable-gui:
	MUJOCO_GL=glfw python -m allostatic_handover.experiments.eval_ppo --backend hrgym --handover-env original --hrgym-wrapper-stack safe_ik --hrgym-shield-type PFL --reward-variant task_only --model-path outputs/ppo_original_handover_actor_critic_stable_20260611/best_model.zip --episodes 3 --horizon 1000 --device cpu --seed 12071 --render --print-step-info --print-interval 25 --output-dir outputs/eval_ppo_original_handover_actor_critic_stable_best_gui

ppo-hrgym-smoke:
	MUJOCO_GL=egl python -m allostatic_handover.experiments.train_ppo --backend hrgym --reward-variant task_only --total-timesteps 128 --horizon 20 --n-steps 32 --batch-size 32 --n-epochs 1 --device cpu --seed 7 --output-dir outputs/ppo_hrgym_headless_smoke

eval-ppo-hrgym-smoke:
	MUJOCO_GL=egl python -m allostatic_handover.experiments.eval_ppo --backend hrgym --reward-variant task_only --model-path outputs/ppo_hrgym_headless_smoke/model_final.zip --episodes 2 --horizon 20 --device cpu --seed 107 --output-dir outputs/eval_ppo_hrgym_headless_smoke --print-step-info --print-interval 10

ppo-allostatic-safe-ik-smoke:
	MUJOCO_GL=egl python -m allostatic_handover.experiments.train_ppo --backend hrgym --handover-env allostatic --hrgym-wrapper-stack safe_ik --hrgym-shield-type PFL --reward-variant task_only --total-timesteps 128 --horizon 20 --n-steps 32 --batch-size 32 --n-epochs 1 --device cpu --seed 101 --output-dir outputs/ppo_allostatic_safe_ik_smoke

eval-ppo-allostatic-safe-ik-smoke:
	MUJOCO_GL=egl python -m allostatic_handover.experiments.eval_ppo --backend hrgym --handover-env allostatic --hrgym-wrapper-stack safe_ik --hrgym-shield-type PFL --reward-variant task_only --model-path outputs/ppo_allostatic_safe_ik_smoke/model_final.zip --episodes 2 --horizon 20 --device cpu --seed 10101 --output-dir outputs/eval_ppo_allostatic_safe_ik_smoke --print-step-info --print-interval 10

ppo-allostatic-compare-safe-ik:
	MUJOCO_GL=egl python -m allostatic_handover.experiments.train_ppo_comparison --backend hrgym --handover-env allostatic --hrgym-wrapper-stack safe_ik_air --hrgym-shield-type PFL --total-timesteps 500000 --eval-episodes 5 --horizon 1000 --seed 101 --device cpu --wandb-mode online --wandb-project allostatic-handover-mvp --wandb-group ppo_allostatic_readiness_hold_eta_air_compare --expert-bc-speech-policy excessive_speech --output-root outputs/ppo_allostatic_readiness_hold_eta_air_compare_seed101

tmux-ppo-allostatic-compare-safe-ik:
	mkdir -p outputs/ppo_allostatic_readiness_hold_eta_air_compare_seed101 && ./.conda/bin/tmux new-session -d -s allostatic_ppo_readiness_hold_eta_air_compare "cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover && MUJOCO_GL=egl ./.conda/bin/python -u -m allostatic_handover.experiments.train_ppo_comparison --backend hrgym --handover-env allostatic --hrgym-wrapper-stack safe_ik_air --hrgym-shield-type PFL --total-timesteps 500000 --eval-episodes 5 --horizon 1000 --seed 101 --device cpu --wandb-mode online --wandb-project allostatic-handover-mvp --wandb-group ppo_allostatic_readiness_hold_eta_air_compare --expert-bc-speech-policy excessive_speech --output-root outputs/ppo_allostatic_readiness_hold_eta_air_compare_seed101 > outputs/ppo_allostatic_readiness_hold_eta_air_compare_seed101/run.log 2>&1"

tail-ppo-allostatic-compare-safe-ik:
	tail -f outputs/ppo_allostatic_readiness_hold_eta_air_compare_seed101/run.log

ppo-original-handover-wandb:
	MUJOCO_GL=egl python -m allostatic_handover.experiments.train_ppo --backend hrgym --handover-env original --reward-variant task_only --total-timesteps 200000 --horizon 1000 --n-steps 64 --batch-size 64 --n-epochs 20 --device cpu --seed 21 --output-dir outputs/ppo_original_handover_wandb --wandb-mode online --wandb-project allostatic-handover-mvp --wandb-group ppo_original_handover --wandb-name ppo_original_handover_wandb

ppo-original-handover-bc-smoke:
	MUJOCO_GL=egl python -m allostatic_handover.experiments.train_ppo --backend hrgym --handover-env original --hrgym-wrapper-stack safe_ik --hrgym-shield-type PFL --reward-variant task_only --expert-bc-rollouts 10 --expert-bc-epochs 300 --expert-bc-batch-size 512 --expert-bc-learning-rate 0.001 --expert-bc-motion-loss-weight 2.0 --expert-bc-gripper-loss-weight 2.0 --expert-bc-action-std 0.05 --expert-dagger-iterations 3 --expert-dagger-rollouts 3 --expert-dagger-epochs 200 --expert-dagger-max-steps 250 --total-timesteps 2048 --horizon 1000 --learning-rate 0.0001 --n-steps 256 --batch-size 256 --n-epochs 3 --device cpu --seed 55 --output-dir outputs/ppo_original_handover_safe_ik_bc_smoke

eval-ppo-original-handover-bc-smoke:
	MUJOCO_GL=egl python -m allostatic_handover.experiments.eval_ppo --backend hrgym --handover-env original --hrgym-wrapper-stack safe_ik --hrgym-shield-type PFL --reward-variant task_only --model-path outputs/ppo_original_handover_safe_ik_bc_smoke/model_final.zip --episodes 3 --horizon 1000 --device cpu --seed 3055 --output-dir outputs/eval_ppo_original_handover_safe_ik_bc_smoke --print-step-info --print-interval 100

test:
	python -m unittest discover -s tests
