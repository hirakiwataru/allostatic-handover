#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/k_iwamoto/sim_data/Projects/allostatic-handover}"
MJLAB_DIR="${MJLAB_DIR:-/mnt/k_iwamoto/sim_data/Projects/mjlab}"
DREAMERV3_DIR="${DREAMERV3_DIR:-/mnt/k_iwamoto/sim_data/Projects/dreamerv3}"

GPU_ID="${GPU_ID:-0}"
NUM_ENVS="${NUM_ENVS:-64}"
MAX_ITER="${MAX_ITER:-500}"
SEED="${SEED:-101}"
SLEEP_SECONDS="${SLEEP_SECONDS:-10}"
WANDB_PROJECT="${WANDB_PROJECT:-allostatic-handover-mjlab}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
WANDB_GROUP="${WANDB_GROUP:-three_condition_compare_${RUN_ID}}"

WM_NUM_ENVS="${WM_NUM_ENVS:-${NUM_ENVS}}"
WM_STEPS="${WM_STEPS:-4096}"
WM_UPDATES="${WM_UPDATES:-5000}"
WM_BATCH_SIZE="${WM_BATCH_SIZE:-64}"
WM_SEQ_LEN="${WM_SEQ_LEN:-64}"
WM_POLICY="${WM_POLICY:-mixed}"

OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/outputs/mjlab_three_condition_compare/${RUN_ID}}"
WORLD_MODEL_DIR="${OUT_DIR}/world_model"
WORLD_MODEL_DATASET="${WORLD_MODEL_DIR}/task_only_speech_dataset.npz"
WORLD_MODEL_BELIEF="${WORLD_MODEL_DIR}/belief_distill.pt"
RUN_LOG="${OUT_DIR}/run.log"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${PROJECT_DIR}/.uvcache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/k_iwamoto/sim_data/tmp/xdg_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/mnt/k_iwamoto/sim_data/tmp/matplotlib}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export WANDB_RUN_GROUP="${WANDB_GROUP}"

mkdir -p "${OUT_DIR}" "${WORLD_MODEL_DIR}"

log() {
  local message="$1"
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${message}" | tee -a "${RUN_LOG}"
}

finish_stage() {
  local stage="$1"
  log "Finished ${stage}; sleeping ${SLEEP_SECONDS}s before next stage."
  sync
  sleep "${SLEEP_SECONDS}"
}

run_stage() {
  local stage="$1"
  local logfile="$2"
  shift 2
  log "Starting ${stage}"
  log "Stage log: ${logfile}"
  {
    printf '[%s] command:' "$(date '+%Y-%m-%d %H:%M:%S')"
    printf ' %q' "$@"
    printf '\n'
    "$@"
  } >"${logfile}" 2>&1
  finish_stage "${stage}"
}

run_stage_shell() {
  local stage="$1"
  local logfile="$2"
  local command="$3"
  log "Starting ${stage}"
  log "Stage log: ${logfile}"
  log "Command: ${command}"
  bash -lc "${command}" >"${logfile}" 2>&1
  finish_stage "${stage}"
}

log "3-condition PPO sequence run_id=${RUN_ID}"
log "out_dir=${OUT_DIR}"
log "wandb_project=${WANDB_PROJECT}"
log "wandb_group=${WANDB_GROUP}"
log "gpu_id=${GPU_ID} num_envs=${NUM_ENVS} max_iter=${MAX_ITER} seed=${SEED}"
log "wm_num_envs=${WM_NUM_ENVS} wm_steps=${WM_STEPS} wm_updates=${WM_UPDATES} wm_policy=${WM_POLICY}"

run_stage "mjlab editable install" "${OUT_DIR}/install.log" \
  make -C "${PROJECT_DIR}" mjlab-install

run_stage_shell "TaskOnlySpeech PPO" "${OUT_DIR}/task_only_speech.log" \
  "cd '${MJLAB_DIR}' && uv run train Mjlab-Allostatic-Handover-Full-TaskOnlySpeech \
    --gpu-ids '[0]' \
    --agent.logger wandb \
    --agent.wandb-project '${WANDB_PROJECT}' \
    --agent.experiment-name allostatic_handover_full_task_only_speech_yam \
    --agent.run-name '${RUN_ID}_task_only_speech_seed${SEED}' \
    --env.scene.num-envs '${NUM_ENVS}' \
    --agent.max-iterations '${MAX_ITER}'"

run_stage_shell "SpeechPenalty PPO" "${OUT_DIR}/speech_penalty.log" \
  "cd '${MJLAB_DIR}' && uv run train Mjlab-Allostatic-Handover-Full-SpeechPenalty \
    --gpu-ids '[0]' \
    --agent.logger wandb \
    --agent.wandb-project '${WANDB_PROJECT}' \
    --agent.experiment-name allostatic_handover_full_speech_penalty_yam \
    --agent.run-name '${RUN_ID}_speech_penalty_seed${SEED}' \
    --env.scene.num-envs '${NUM_ENVS}' \
    --agent.max-iterations '${MAX_ITER}'"

run_stage_shell "world-model dataset collection" "${OUT_DIR}/world_model_dataset.log" \
  "cd '${MJLAB_DIR}' && uv run python '${PROJECT_DIR}/scripts/collect_mjlab_world_model_dataset.py' \
    --output '${WORLD_MODEL_DATASET}' \
    --num-envs '${WM_NUM_ENVS}' \
    --steps '${WM_STEPS}' \
    --seed '${SEED}' \
    --device cuda:0 \
    --policy '${WM_POLICY}' \
    --wandb-mode online \
    --wandb-project '${WANDB_PROJECT}' \
    --wandb-group '${WANDB_GROUP}' \
    --wandb-run-name '${RUN_ID}_world_model_dataset_${WM_POLICY}_seed${SEED}'"

run_stage_shell "belief world-model training" "${OUT_DIR}/world_model_train.log" \
  "cd '${MJLAB_DIR}' && PYTHONPATH='${DREAMERV3_DIR}:${PROJECT_DIR}' uv run python '${PROJECT_DIR}/scripts/train_dreamer_world_model.py' \
    --dataset '${WORLD_MODEL_DATASET}' \
    --output-dir '${WORLD_MODEL_DIR}' \
    --updates '${WM_UPDATES}' \
    --batch-size '${WM_BATCH_SIZE}' \
    --seq-len '${WM_SEQ_LEN}' \
    --device cuda:0 \
    --dreamerv3-path '${DREAMERV3_DIR}' \
    --wandb-mode online \
    --wandb-project '${WANDB_PROJECT}' \
    --wandb-run-name '${RUN_ID}_world_model_${WM_POLICY}_seed${SEED}'"

if [[ ! -f "${WORLD_MODEL_BELIEF}" ]]; then
  log "ERROR: missing belief model ${WORLD_MODEL_BELIEF}"
  exit 1
fi

run_stage_shell "AllostaticBelief PPO" "${OUT_DIR}/allostatic_belief.log" \
  "cd '${MJLAB_DIR}' && ALLOSTATIC_WM_BELIEF_MODEL='${WORLD_MODEL_BELIEF}' uv run train Mjlab-Allostatic-Handover-Full-AllostaticBelief \
    --gpu-ids '[0]' \
    --agent.logger wandb \
    --agent.wandb-project '${WANDB_PROJECT}' \
    --agent.experiment-name allostatic_handover_full_allostatic_belief_yam \
    --agent.run-name '${RUN_ID}_allostatic_belief_seed${SEED}' \
    --env.scene.num-envs '${NUM_ENVS}' \
    --agent.max-iterations '${MAX_ITER}'"

log "All stages completed successfully."
log "World model belief: ${WORLD_MODEL_BELIEF}"
