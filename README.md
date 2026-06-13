# allostatic-handover

`human-robot-gym` の `RobotHumanHandoverCart` を直接変更せず、外部リポジトリ側で allostatic load、speech token、人間 hidden state FSM、報酬切替、ログ、可視化GUIを追加するMVPです。

## Repository Layout

- `allostatic_handover/envs/`: allostatic load、human FSM、speech token、`AllostaticRobotHumanHandoverCart`
- `allostatic_handover/policies/`: scripted policies
- `allostatic_handover/experiments/`: rollout、PPO、退化方策比較
- `allostatic_handover/logging/`: local CSV/JSONL と optional W&B
- `allostatic_handover/dashboard/`: rollout可視化GUI
- `allostatic_handover/mjlab_tasks/`: Mjlab外部タスク `Mjlab-Allostatic-Handover-Yam` / `Mjlab-Allostatic-Handover-Full`
- `configs/`: reward / speech / human state / W&B 設定例

## Fresh Setup From GitHub

このrepoは `human-robot-gym`、`mjlab`、必要に応じて `dreamerv3` を外部依存として参照します。これらの本体リポジトリはこのrepoには含めません。以下は `/mnt/k_iwamoto/sim_data/Projects` と同じ sibling layout を作る例です。

```bash
export PROJECTS=/mnt/k_iwamoto/sim_data/Projects
mkdir -p "$PROJECTS"
cd "$PROJECTS"

git clone https://github.com/mujocolab/mjlab.git
git clone https://github.com/TUMcps/human-robot-gym.git
git clone https://github.com/danijar/dreamerv3.git
git clone https://github.com/hirakiwataru/allostatic-handover.git
```

Mjlab側の環境を作成します。Mjlabの依存は上流のREADMEに従って用意してください。このプロジェクト側では、作成済みの `mjlab/.venv` に外部task packageとして editable install します。

```bash
cd "$PROJECTS/mjlab"
uv sync

cd "$PROJECTS/allostatic-handover"
UV_CACHE_DIR="$PROJECTS/allostatic-handover/.uvcache" \
  uv pip install --python "$PROJECTS/mjlab/.venv/bin/python" \
  --no-deps --no-build-isolation -e .
```

`human-robot-gym` ベースの旧PPO/評価スクリプトを使う場合は、別途 `environment.yml` からconda環境を作れます。

```bash
cd "$PROJECTS/allostatic-handover"
conda env create -f environment.yml
conda activate allostatic-handover
```

Full Mjlab taskはHRGymの人体XML/mesh/animation metadataをローカルコピーして使います。ライセンス確認前のvendor assetはgit管理しない方針なので、clone後に各自のローカル `human-robot-gym` からコピーしてください。

```bash
cd "$PROJECTS/allostatic-handover"
python3 scripts/copy_hrgym_assets.py --hrgym-root "$PROJECTS/human-robot-gym" --include-full-handover-assets
```

task登録確認:

```bash
cd "$PROJECTS/mjlab"
UV_CACHE_DIR="$PROJECTS/allostatic-handover/.uvcache" \
  MPLCONFIGDIR="$PROJECTS/../tmp/matplotlib" \
  uv run list-envs | grep Allostatic
```

headless smoke:

```bash
cd "$PROJECTS/allostatic-handover"
make mjlab-install
MUJOCO_GL=egl make mjlab-train-full-speech-penalty-smoke
```

Linuxモニタで可視化する場合は、この環境では `DISPLAY=:1` と `XAUTHORITY=/run/user/1000/gdm/Xauthority` を使っています。別環境では適宜上書きしてください。

```bash
cd "$PROJECTS/allostatic-handover"
make mjlab-play-full-task-only-speech MJLAB_DISPLAY=:1 MJLAB_XAUTHORITY=/run/user/1000/gdm/Xauthority
```

## PPO Comparison Conditions

現在の主比較条件は次の3つです。いずれも `Mjlab-Allostatic-Handover-Full` 系で、YAMロボット、HRGym風human/table/hammer配置、5D action `[dx, dy, dz, gripper, speech_scalar]` を使います。

| Condition | Task ID | Actor/Critic observation | Reward additions | Intended behavior |
|---|---|---|---|---|
| TaskOnlySpeech | `Mjlab-Allostatic-Handover-Full-TaskOnlySpeech` | 公開観測のみ。真のFSM/readiness/loadと手書きproxyなし | なし | 発話を自由に使い、成功率を最大化する |
| SpeechPenalty | `Mjlab-Allostatic-Handover-Full-SpeechPenalty` | TaskOnlySpeechと同じ公開観測のみ | 隠れspeech loadの指数型ペナルティ `-0.04` | 発話頻度を負荷が蓄積しない範囲に抑える |
| AllostaticBelief | `Mjlab-Allostatic-Handover-Full-AllostaticBelief` | 公開観測 + frozen world-model belief。真値/proxyなし | 指数型speech penalty `-0.02`、load penalty `-0.05`、waiting penalty `-0.10` | beliefで隠れ状態を推定し、成功率と負荷低減を両立する |

3条件ともactorとcriticの両方から真の `human_state_id`、`human_readiness`、`allostatic_load_total`、`readiness_belief`、`load_proxy` を外しています。hidden loadは観測には入りませんが、ログ、readiness decay、FSM遷移には使われます。FSM遷移の閾値は共通で `overload_threshold=7.0`、`withdrawal_threshold=9.0` です。

共通action:

```text
dx, dy, dz      Yam grasp_site の相対IK制御
gripper         Yam left_finger の開閉
speech_scalar   [-1, 1] を RobotSpeechToken に離散化
```

発話token:

```text
SILENCE
ANNOUNCE_HANDOVER  今から渡します
ASK_READY          準備できましたか
REASSURE           ゆっくりで大丈夫です
SAY_WAITING        待ちます
SAY_RELEASING      離します
ASK_CONFIRMATION   取れましたか
```

共通公開観測:

```text
joint_pos, joint_vel
ee_to_cube, ee_to_hand, cube_to_hand
speech_context = last_speech_token, previous_speech_token, repeated_flag
phase_progress = phase, reach_progress, retreat_progress
actions
```

共通task reward:

```text
handover_precise      0.4
success              30.0
robot_grasp_approach  1.2
robot_grasp           0.15
carry_to_hand         0.25
release_at_hand       5.0
handoff              25.0
time_penalty         -0.02
action_rate_l2       -0.002
joint_pos_limits     -5.0
```

発話頻度ペナルティは `SpeechPenalty` と `AllostaticBelief` で同じ関数です。

```python
speech_load = attention_load + turn_taking_load
excess = clamp(speech_load - 0.8, min=0.0, max=4.0)
penalty = expm1(excess * 0.5)
```

学習開始コマンド:

```bash
cd "$PROJECTS/allostatic-handover"

make tmux-mjlab-train-full-task-only-speech-wandb-gpu
make tmux-mjlab-train-full-speech-penalty-wandb-gpu

# AllostaticBeliefは先にworld model/belief_distill.ptを作る
make tmux-dreamer-train-world-model
make tmux-mjlab-train-full-allostatic-belief-wandb-gpu \
  WM_MODEL="$PROJECTS/allostatic-handover/outputs/world_model/full/belief_distill.pt"
```

## Mjlab External Task

`Mjlab-Allostatic-Handover-Yam` は Mjlab の `Mjlab-Lift-Cube-Yam` をベースにした簡易手渡しタスクです。`human-robot-gym` と `mjlab` 本体は変更せず、このrepoを Mjlab 環境へ editable install して task registry に追加します。

実装内容:

- action は `[dx, dy, dz, gripper, speech_scalar]`
- `dx, dy, dz` は Yam の `grasp_site` に対する `DifferentialIKActionCfg`
- `gripper` は Yam の `left_finger` position target
- `speech_scalar` は `RobotSpeechToken` に変換し、readiness / phase / allostatic load を更新
- 人側は primitive の胴体・腕・手ターゲットで表現
- phase は `APPROACH -> REACH_OUT -> RETREAT -> COMPLETE`
- reward variant は `task_only`, `speech_penalty`, `allostatic`

`Mjlab-Allostatic-Handover-Full` は同じ Yam action stack を使いながら、HRGym `RobotHumanHandoverCart` の人体 XML、RobotHumanHandover pkl animation、HRGym 風 table、hammer-like object、手渡し配置へ寄せた task です。人のroot poseはHRGymの基準回転 `Rotation.from_quat([0.5, 0.5, 0.5, 0.5])` と同じ変換を使います。ロボットはSchunk完全再現ではなくYamを維持し、Yam base をテーブル面に置きます。Yam は人側へ向く yaw `0` の姿勢で、hammer は Yam base 直下ではなく `grasp_site` 前方の少し離れた到達可能範囲に机上配置されます。hammer yaw は `pi/2` 固定で、人に対して横向きになるようにしています。config上のroot zとHRGym table topはいずれも `0.845` です。Full task は vendor asset が必要です。

Full task 系の使い分け:

- `Mjlab-Allostatic-Handover-Full`: speech / allostatic あり、Yamはテーブル面、物体は机上スタート
- `Mjlab-Allostatic-Handover-Full-TaskOnly`: task-only PPO用、Yamはテーブル面、物体は机上スタート
- `Mjlab-Allostatic-Handover-Full-TaskOnlySpeech`: task-only rewardだがspeech actionとreadiness FSMは有効。正しい手渡しcueで人が準備状態になり、HRGym animationが歩み寄り/手差しへ進む比較用条件。actor/criticともFSM/readiness/load真値や手書きproxyは見ない。loadは報酬罰には使わないが、readiness decayとFSM overload/withdrawal遷移には使う
- `Mjlab-Allostatic-Handover-Full-SpeechPenalty`: `TaskOnlySpeech` と同じ5D action / FSM / 配置で、actor/criticにはFSM/readiness/load真値や手書きproxyを見せず、発話頻度で蓄積する隠れ負荷に対して指数型ペナルティを追加する比較用条件
- `Mjlab-Allostatic-Handover-Full-AllostaticBelief`: `TaskOnlySpeech` と同じ5D action / FSM / 配置で、actor/critic観測から真のFSM/readiness/loadと手書きproxyを外し、凍結world modelのbelief出力だけを追加する比較用条件。報酬はtask shapingに指数型 `speech_penalty=-0.02`, `allostatic_load=-0.05`, `waiting_cost=-0.10` を追加
- `Mjlab-Allostatic-Handover-Full-TaskOnly-GraspedStart`: 旧checkpoint再生用のcurriculum条件。Yamは高い台座上、物体は初期把持済み

Full task 用 asset copy:

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
make copy-hrgym-full-assets
```

Mjlab 環境へ外部 task を登録:

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
/mnt/k_iwamoto/sim_data/Projects/mjlab/.venv/bin/python -m pip install -e .
```

この Linux 環境の Mjlab `.venv` には `pip` module が無いため、実際には次の `uv` コマンドで登録確認済みです。

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
UV_CACHE_DIR=/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/.uvcache \
  uv pip install --python /mnt/k_iwamoto/sim_data/Projects/mjlab/.venv/bin/python \
  --no-deps --no-build-isolation -e .
```

task登録確認:

```bash
cd /mnt/k_iwamoto/sim_data/Projects/mjlab
UV_CACHE_DIR=/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/.uvcache \
  MPLCONFIGDIR=/mnt/k_iwamoto/sim_data/tmp/matplotlib \
  uv run list-envs | grep Allostatic
```

Full task のHRGym/Mjlab相対配置をJSONで比較:

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
make copy-hrgym-full-assets
make mjlab-install
make mjlab-check-full-layout
```

Linuxモニタで native viewer:

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
make mjlab-check-display
make mjlab-play-yam
```

このLinux PCでは現状 `DISPLAY=:1` と `XAUTHORITY=/run/user/1000/gdm/Xauthority` が必要です。`Makefile` のデフォルトもそれに合わせています。別のdisplayを使う場合は次のように上書きします。

```bash
make mjlab-play-yam MJLAB_DISPLAY=:0 MJLAB_XAUTHORITY="$XAUTHORITY"
```

`make mjlab-play-*` 系ターゲットは `scripts/run_mjlab_play.py` 経由で起動します。viewerを止める時は通常どおり terminal で `Ctrl+C` を押してください。`uv run play` と実体の Python viewer は同じプロセスグループとしてまとめて終了されるため、MuJoCo viewer の残プロセスが残りにくくなっています。

Full task の通常確認用環境を Linux モニタで native viewer:

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
make copy-hrgym-full-assets
make mjlab-install
make mjlab-check-display
make mjlab-play-full
```

`make mjlab-play-full` は、混線を避けるため `Mjlab-Allostatic-Handover-Full-TaskOnly` を開きます。これは Yam がテーブル面にあり、物体が机上から開始し、初期把持しない通常確認用の環境です。同じ環境を明示的に指定する場合:

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
make copy-hrgym-full-assets
make mjlab-install
make mjlab-check-display
make mjlab-play-full-task-only
```

学習済み方策を確認する場合はcheckpointを明示します。成功やタイムアウトでepisodeが切り替わる瞬間は、前episodeの終端姿勢から次episodeの初期姿勢へMuJoCo状態がリセットされるため、ハンマーが瞬時に初期姿勢へ戻ります。このreset境界のジャンプを見ずに1 episodeの挙動だけ確認したい場合は、terminationを無効にしたno-reset確認用ターゲットを使います。

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
make mjlab-install
make mjlab-play-full-task-only-trained-no-reset \
  CKPT=/mnt/k_iwamoto/sim_data/Projects/mjlab/logs/rsl_rl/allostatic_handover_full_task_only_yam/2026-06-13_12-22-53_ppo_task_only_fixed_animation_object_grasp_shaping_v2_64env_500iter/model_499.pt
```

episode境界のresetも含めて通常の評価ループを見たい場合:

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
make mjlab-install
make mjlab-play-full-task-only-trained \
  CKPT=/mnt/k_iwamoto/sim_data/Projects/mjlab/logs/rsl_rl/allostatic_handover_full_task_only_yam/2026-06-13_12-22-53_ppo_task_only_fixed_animation_object_grasp_shaping_v2_64env_500iter/model_499.pt
```

表示が想定と違う場合は、viewerを開く前に次で実際に読み込まれているtask/root高さと物体位置を確認します。正しければ `task_id=Mjlab-Allostatic-Handover-Full-TaskOnly`、`reset.robot_root=(..., ..., 0.845000)`、`reset.grasp_site` はYam baseより前方、`reset.object_root` はYam base直下ではなく `grasp_site` 近傍、`reset.table_top=(..., ..., 0.845000)`、`action_dim=4` が出ます。

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
make mjlab-install
make mjlab-diagnose-full-task-only
```

speech / allostatic ありのFull環境を確認する場合だけ、こちらを使います。

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
make mjlab-play-full-allostatic
```

speech actionありのtask-only比較環境を確認する場合はこちらです。この環境は5次元action `[dx, dy, dz, gripper, speech_scalar]` で、`ANNOUNCE_HANDOVER`（「今から渡します」）などのcueによりreadinessが閾値を超えるまで、人のFull animation開始を止めます。報酬はtask-onlyで、speech/load/waiting penaltyは入りません。actor/criticとも、人のFSM/readiness/load真値や手書きproxyは観測しません。

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
make mjlab-play-full-task-only-speech
```

speech penalty比較環境を確認する場合はこちらです。`TaskOnlySpeech` と同じ環境ですが、actor/critic観測から `readiness_belief` / `load_proxy` / privileged hidden stateを外し、発話頻度で蓄積する隠れspeech loadが閾値を超えた時だけ `expm1` 型の滑らかなペナルティを報酬から引きます。allostatic load / waiting penalty は入りません。loadは `TaskOnlySpeech` と同じくreadiness decayとFSM overload/withdrawal遷移には使われます。

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
make mjlab-play-full-speech-penalty
```

AllostaticBelief比較環境は、先にworld modelから `belief_distill.pt` を作ります。dataset収集では公開観測 `public_obs` と5D actionだけをworld model入力として保存し、真の `human_state_id`, `human_readiness`, `allostatic_load_total`, `phase`, `reach_progress` は教師ラベルとしてだけ保存します。PPO actor/criticにはこれらの真値や `readiness_belief` / `load_proxy` は入らず、world model belief出力だけが追加されます。発話罰は `SpeechPenalty` と同じ指数型speech load penaltyです。

smokeを一通り確認する場合:

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
make mjlab-collect-world-model-dataset-smoke
make dreamer-train-world-model-smoke
make mjlab-train-full-allostatic-belief-smoke
```

smokeの出力先:

- dataset: `outputs/world_model/latest/task_only_speech_dataset.npz`
- world model source checkpoint: `outputs/world_model/latest/world_model.ckpt`
- PPO用の凍結PyTorch belief estimator: `outputs/world_model/latest/belief_distill.pt`
- normalization: `outputs/world_model/latest/normalization.json`
- world model metrics: `outputs/world_model/latest/metrics.jsonl`

Exact DreamerV3/RSSMを使う場合は、DreamerV3 cloneを変更せず、専用venvから `dreamerv3.agent.Agent` / `dreamerv3.rssm.RSSM` / `Encoder` / `Decoder` をそのまま使います。真の `human_state_id`, `human_readiness`, `allostatic_load_total` はDreamerV3 encoderの観測には入れず、`ext_space` の補助教師ラベルとしてだけ使います。

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
make dreamerv3-exact-venv
make dreamerv3-exact-check-deps
make dreamerv3-exact-check-jax-cpu
make dreamer-train-world-model-exact-smoke
```

Exact smokeの出力先:

- exact DreamerV3 checkpoint: `outputs/world_model/dreamerv3_exact_smoke/world_model.ckpt`
- exact DreamerV3 metrics: `outputs/world_model/dreamerv3_exact_smoke/metrics.jsonl`
- exact DreamerV3 config: `outputs/world_model/dreamerv3_exact_smoke/config.yaml`

JAX/CUDAが使えるかを切り分ける場合:

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
make dreamerv3-exact-check-jax-gpu
make dreamer-train-world-model-exact-gpu-smoke
```

このLinux環境では、Mjlab/PPOのCUDAは動作していますが、JAX GPU runtimeは最小の `jax.numpy` 行列積でもsegmentation faultすることがあります。その場合はDreamerV3 exact smokeをCPUで通し、PPO runtimeには既存の `belief_distill.pt` を使います。RSL-RL/MjlabはPyTorchなので、学習済みDreamerV3 posteriorを毎step JAXで呼ばず、凍結PyTorch belief estimatorをPPO観測へ入れる構成にしています。

現在の人FSMはDreamerV3で学習可能な対象です。遷移は、公開観測に含めているphase/reach progress、ロボット・物体・手ターゲットの相対量、5D actionの `speech_scalar`、および発話履歴から推定できる低次元状態です。ただしdatasetには、沈黙、正しい呼びかけ、発話過多、保持時間切れ、handoff成功/失敗のtrajectoryを混ぜ、readiness hold/decayを十分観測させる必要があります。

本格world model学習をtmuxで開始する場合:

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
make tmux-dreamer-train-world-model
make tail-dreamer-world-model
```

この本格world model runは `wandb-mode online` で、`world_model/human_state_acc`, `world_model/readiness_mae`, `world_model/load_mae` などを `allostatic-handover-mjlab` projectへ送ります。smoke targetはローカル成果物のみを作ります。

AllostaticBelief PPOをW&B付きGPU学習で開始する場合:

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
make tmux-mjlab-train-full-allostatic-belief-wandb-gpu \
  WM_MODEL=/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/outputs/world_model/full/belief_distill.pt
make tail-mjlab-full-allostatic-belief
```

AllostaticBelief policyを評価する場合:

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
make mjlab-eval-full-allostatic-belief \
  CKPT=/path/to/model.pt \
  WM_MODEL=/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/outputs/world_model/full/belief_distill.pt
```

AllostaticBelief policyをLinuxモニタで可視化する場合:

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
make mjlab-play-full-allostatic-belief-trained \
  CKPT=/path/to/model.pt \
  WM_MODEL=/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/outputs/world_model/full/belief_distill.pt
```

発話ありTaskOnlyの学習済みpolicyを、MuJoCo native viewerとライブ診断GUIつきで可視化する場合:

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
make mjlab-install
make mjlab-play-full-task-only-speech-trained-live \
  CKPT=/mnt/k_iwamoto/sim_data/Projects/mjlab/logs/rsl_rl/allostatic_handover_full_task_only_speech_yam/2026-06-13_15-03-59_ppo_task_only_speech_fixed_animation_object_64env_500iter/model_499.pt
```

別ブラウザで `http://127.0.0.1:7860/live.html` を開くと、発話履歴、人のhidden state、readiness/load、palm distance、handoff判定gateを確認できます。Linux外のPCから見る場合は `127.0.0.1` をLinux PCのIPアドレスに置き換えます。ログは `outputs/mjlab_live/*/live.jsonl` に保存されます。

旧 `ppo_task_only_reward_rescale_64env` checkpoint は `GraspedStart` 条件で学習したものです。可視化する場合は task ID も `GraspedStart` に合わせます。

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
make mjlab-play-full-task-only-grasped-start \
  CKPT=/mnt/k_iwamoto/sim_data/Projects/mjlab/logs/rsl_rl/allostatic_handover_full_task_only_yam/2026-06-12_18-01-20_ppo_task_only_reward_rescale_64env/model_999.pt
```

headless smoke PPO。Mjlab 学習は GPU を使う前提で、GPU 0 を指定します。

```bash
cd /mnt/k_iwamoto/sim_data/Projects/mjlab
UV_CACHE_DIR=/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/.uvcache \
  XDG_CACHE_HOME=/mnt/k_iwamoto/sim_data/tmp/xdg_cache \
  MPLCONFIGDIR=/mnt/k_iwamoto/sim_data/tmp/matplotlib \
  CUDA_VISIBLE_DEVICES=0 \
  MUJOCO_GL=egl \
  uv run train Mjlab-Allostatic-Handover-Yam \
  --gpu-ids '[0]' \
  --agent.logger tensorboard \
  --env.scene.num-envs 64 \
  --agent.max-iterations 2
```

GPU が見えない Codex sandbox などで配管だけ確認したい場合の fallback:

```bash
cd /mnt/k_iwamoto/sim_data/Projects/mjlab
UV_CACHE_DIR=/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/.uvcache \
  XDG_CACHE_HOME=/mnt/k_iwamoto/sim_data/tmp/xdg_cache \
  MPLCONFIGDIR=/mnt/k_iwamoto/sim_data/tmp/matplotlib \
  MUJOCO_GL=egl \
  uv run train Mjlab-Allostatic-Handover-Yam \
  --gpu-ids None \
  --agent.logger tensorboard \
  --env.scene.num-envs 64 \
  --agent.max-iterations 2
```

W&B にログを送る GPU 学習:

```bash
cd /mnt/k_iwamoto/sim_data/Projects/mjlab
UV_CACHE_DIR=/mnt/k_iwamoto/sim_data/Projects/allostatic-handover/.uvcache \
  XDG_CACHE_HOME=/mnt/k_iwamoto/sim_data/tmp/xdg_cache \
  MPLCONFIGDIR=/mnt/k_iwamoto/sim_data/tmp/matplotlib \
  CUDA_VISIBLE_DEVICES=0 \
  MUJOCO_GL=egl \
  uv run train Mjlab-Allostatic-Handover-Yam \
  --gpu-ids '[0]' \
  --agent.logger wandb \
  --agent.wandb-project allostatic-handover-mjlab \
  --agent.experiment-name allostatic_handover_yam \
  --env.scene.num-envs 1024 \
  --agent.max-iterations 5000
```

Full task の headless GPU smoke:

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
make copy-hrgym-full-assets
make mjlab-install
make mjlab-train-full-smoke
```

Full task-only 環境で、学習初期の探索を大きめにして動画確認する場合:

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
make copy-hrgym-full-assets
make mjlab-install
make mjlab-train-full-task-only-visual-explore
```

このターゲットは可視化デバッグ用です。通常設定の `delta_pos_scale=0.045`, `max_dq=0.12`, 初期std `1.0` では、1 env / 10 iter の初期方策動画だとYamの動きがかなり小さく見えます。`visual-explore` では一時的に `delta_pos_scale=0.12`, `max_dq=0.25`, 初期std `2.0` に上げ、探索で腕が動くかを確認します。本格学習にそのまま使う設定ではありません。

speech actionありtask-only条件の短時間smoke:

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
make copy-hrgym-full-assets
make mjlab-install
make mjlab-train-full-task-only-speech-smoke
```

speech actionありtask-only条件をW&Bへ送って500 iter学習:

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
make copy-hrgym-full-assets
make mjlab-install
make tmux-mjlab-train-full-task-only-speech-wandb-gpu
make tail-mjlab-full-task-only-speech
```

speech penalty条件をW&Bへ送って500 iter学習:

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
make copy-hrgym-full-assets
make mjlab-install
make tmux-mjlab-train-full-speech-penalty-wandb-gpu
make tail-mjlab-full-speech-penalty
```

学習済みpolicyをheadless評価:

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
make mjlab-eval-full-task-only-speech \
  CKPT=/mnt/k_iwamoto/sim_data/Projects/mjlab/logs/rsl_rl/allostatic_handover_full_task_only_speech_yam/<run>/model_499.pt
```

speech penalty条件のpolicyをheadless評価:

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
make mjlab-eval-full-speech-penalty \
  CKPT=/mnt/k_iwamoto/sim_data/Projects/mjlab/logs/rsl_rl/allostatic_handover_full_speech_penalty_yam/<run>/model_499.pt
```

学習済みpolicyをLinux viewerで可視化:

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
make mjlab-play-full-task-only-speech-trained \
  CKPT=/mnt/k_iwamoto/sim_data/Projects/mjlab/logs/rsl_rl/allostatic_handover_full_task_only_speech_yam/<run>/model_499.pt
```

speech penalty条件のpolicyをLinux viewerで可視化:

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
make mjlab-play-full-speech-penalty-trained \
  CKPT=/mnt/k_iwamoto/sim_data/Projects/mjlab/logs/rsl_rl/allostatic_handover_full_speech_penalty_yam/<run>/model_499.pt
```

Full task を W&B にログ送信して GPU 学習:

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
make copy-hrgym-full-assets
make mjlab-install
make mjlab-train-full-wandb-gpu
```

reward variant の切り替え例:

```bash
MUJOCO_GL=egl uv run train Mjlab-Allostatic-Handover-Yam \
  --gpu-ids '[0]' \
  --agent.logger tensorboard \
  --env.commands.handover.reward-variant task_only \
  --env.scene.num-envs 64 \
  --agent.max-iterations 2
```

human-robot-gym の animation metadata だけを後続の full-fidelity 移植用にコピーする場合:

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
python3 scripts/copy_hrgym_assets.py
```

コピー先の `assets/vendor/human_robot_gym/` はライセンス確認まで git 追跡対象外です。詳細は `ASSET_PROVENANCE.md` を参照してください。

Full task 用の単体テストと visual check:

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
make copy-hrgym-full-assets
MUJOCO_GL=egl python3 -m unittest discover -s tests
```

visual check は `outputs/visual_checks/mjlab_full/` に `reset.png`、`reach_out.png`、`handoff.png` を保存します。

## Conda Setup

`human-robot-gym` 本体が Python 3.13 前提なので、実環境実験は新しい conda env で実行します。

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
conda env create -p ./.conda -f environment.yml
conda activate /mnt/k_iwamoto/sim_data/Projects/allostatic-handover/.conda
```

既存 env に入れる場合:

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
python -m pip install -e ../human-robot-gym[training]
python -m pip install -e .[training]
```

## Sandbox Smoke Test

MuJoCo / robosuite が無い環境でも、mock backend で load/FSM/log/GUI の配管を確認できます。

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
python -m allostatic_handover.experiments.run_scripted_rollouts \
  --backend mock \
  --policy excessive_speech \
  --reward-variant task_only \
  --episodes 2 \
  --horizon 80 \
  --output-dir outputs/smoke_excessive
```

比較をまとめて回す場合:

```bash
python -m allostatic_handover.experiments.eval_degenerate_policy \
  --backend mock \
  --episodes 4 \
  --horizon 120 \
  --output-root outputs/mock_degeneracy_eval
```

## human-robot-gym Backend

実際の `RobotHumanHandoverCart` 拡張を使う場合:

```bash
conda activate allostatic-handover
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
python -m allostatic_handover.experiments.run_scripted_rollouts \
  --backend hrgym \
  --policy minimal_speech \
  --reward-variant task_only \
  --episodes 3 \
  --horizon 1000 \
  --output-dir outputs/hrgym_minimal_task_only
```

allostatic reward 条件:

```bash
python -m allostatic_handover.experiments.run_scripted_rollouts \
  --backend hrgym \
  --policy allostatic_aware \
  --reward-variant allostatic \
  --episodes 3 \
  --horizon 1000 \
  --output-dir outputs/hrgym_allostatic_aware
```

## PPO

PPO は humanoid-bench の `ppo/run_sb3_ppo.py` と同じく、Gymnasium 環境を `Monitor` で包み、Stable-Baselines3 の `PPO("MlpPolicy", ...)` で学習します。`human-robot-gym` / robosuite はそのままだと Gymnasium 環境ではないため、このrepo側の `RobosuiteGymnasiumAdapter` で observation をflatな `Box` に変換します。

`--backend hrgym` では `--handover-env` で環境を切り替えられます。

- `--handover-env original`: human-robot-gym 元実装の `RobotHumanHandoverCart`。発話actionなし。
- `--handover-env allostatic`: `AllostaticRobotHumanHandoverCart`。action最後の1次元を発話scalarとして扱い、hidden state / allostatic load / reward variantを使う。

`--backend hrgym --handover-env original` では `--hrgym-wrapper-stack` で human-robot-gym の学習用 wrapper stack を選べます。

- `raw`: robosuite の joint/action をそのまま SB3 に渡す診断用。
- `safe_ik`: `GymWrapper -> CollisionPreventionWrapper -> IKPositionDeltaWrapper` 相当。action は `(dx, dy, dz, gripper)`。
- `safe_ik_air`: `safe_ik` に action-based expert imitation reward を加えたもの。`RHH-AIR.yaml` 相当の `alpha=0.25`, `beta=0.7` がデフォルト。

`--backend hrgym --handover-env allostatic --hrgym-wrapper-stack safe_ik` では、PPO action は `(dx, dy, dz, gripper, speech_scalar)` です。最後の1次元だけをこのrepo側の wrapper が speech token に変換し、MuJoCo / human-robot-gym には既存と同じ4次元 safe-IK motor action を渡します。

mock backend で配管確認:

```bash
python -m allostatic_handover.experiments.train_ppo \
  --backend mock \
  --reward-variant task_only \
  --total-timesteps 10000 \
  --output-dir outputs/ppo_mock_task_only
```

MuJoCo / human-robot-gym 環境でのheadless smoke学習:

```bash
MUJOCO_GL=egl python -m allostatic_handover.experiments.train_ppo \
  --backend hrgym \
  --handover-env allostatic \
  --reward-variant task_only \
  --total-timesteps 128 \
  --horizon 20 \
  --n-steps 32 \
  --batch-size 32 \
  --n-epochs 1 \
  --device cpu \
  --seed 7 \
  --output-dir outputs/ppo_hrgym_headless_smoke
```

保存済みPPOモデルのheadless評価:

```bash
MUJOCO_GL=egl python -m allostatic_handover.experiments.eval_ppo \
  --backend hrgym \
  --handover-env allostatic \
  --reward-variant task_only \
  --model-path outputs/ppo_hrgym_headless_smoke/model_final.zip \
  --episodes 2 \
  --horizon 20 \
  --device cpu \
  --seed 107 \
  --output-dir outputs/eval_ppo_hrgym_headless_smoke \
  --print-step-info \
  --print-interval 10
```

短縮コマンド:

```bash
make ppo-hrgym-smoke
make eval-ppo-hrgym-smoke
```

allostatic版を `safe_ik` で確認するheadless smoke:

```bash
MUJOCO_GL=egl python -m allostatic_handover.experiments.train_ppo \
  --backend hrgym \
  --handover-env allostatic \
  --hrgym-wrapper-stack safe_ik \
  --hrgym-shield-type PFL \
  --reward-variant task_only \
  --total-timesteps 128 \
  --horizon 20 \
  --n-steps 32 \
  --batch-size 32 \
  --n-epochs 1 \
  --device cpu \
  --seed 101 \
  --output-dir outputs/ppo_allostatic_safe_ik_smoke
```

保存済みsmoke policyのheadless評価:

```bash
MUJOCO_GL=egl python -m allostatic_handover.experiments.eval_ppo \
  --backend hrgym \
  --handover-env allostatic \
  --hrgym-wrapper-stack safe_ik \
  --hrgym-shield-type PFL \
  --reward-variant task_only \
  --model-path outputs/ppo_allostatic_safe_ik_smoke/model_final.zip \
  --episodes 2 \
  --horizon 20 \
  --device cpu \
  --seed 10101 \
  --output-dir outputs/eval_ppo_allostatic_safe_ik_smoke \
  --print-step-info \
  --print-interval 10
```

短縮コマンド:

```bash
make ppo-allostatic-safe-ik-smoke
make eval-ppo-allostatic-safe-ik-smoke
```

元のhuman-robot-gym handover環境でPPOをW&B onlineに流す例:

```bash
MUJOCO_GL=egl python -m allostatic_handover.experiments.train_ppo \
  --backend hrgym \
  --handover-env original \
  --hrgym-wrapper-stack safe_ik \
  --reward-variant task_only \
  --total-timesteps 200000 \
  --horizon 1000 \
  --n-steps 64 \
  --batch-size 64 \
  --n-epochs 20 \
  --device cpu \
  --seed 21 \
  --output-dir outputs/ppo_original_handover_wandb \
  --wandb-mode online \
  --wandb-project allostatic-handover-mvp \
  --wandb-group ppo_original_handover \
  --wandb-name ppo_original_handover_wandb
```

短縮コマンド:

```bash
make ppo-original-handover-wandb
```

expert imitation reward を使う短い確認:

```bash
MUJOCO_GL=egl python -m allostatic_handover.experiments.train_ppo \
  --backend hrgym \
  --handover-env original \
  --hrgym-wrapper-stack safe_ik_air \
  --reward-variant task_only \
  --expert-imitation-alpha 0.25 \
  --expert-imitation-beta 0.7 \
  --total-timesteps 8192 \
  --horizon 1000 \
  --learning-rate 3e-4 \
  --n-steps 256 \
  --batch-size 256 \
  --n-epochs 10 \
  --device cpu \
  --seed 43 \
  --output-dir outputs/ppo_original_handover_safe_ik_air_smoke
```

短時間で「手渡し動作に向かう PPO policy」を確認したい場合は、human-robot-gym の `PickPlaceHumanCartExpert` rollout で actor を行動クローニング初期化してから PPO を回します。

```bash
MUJOCO_GL=egl python -m allostatic_handover.experiments.train_ppo \
  --backend hrgym \
  --handover-env original \
  --hrgym-wrapper-stack safe_ik \
  --hrgym-shield-type PFL \
  --reward-variant task_only \
  --expert-bc-rollouts 10 \
  --expert-bc-epochs 300 \
  --expert-bc-batch-size 512 \
  --expert-bc-learning-rate 0.001 \
  --expert-bc-motion-loss-weight 2.0 \
  --expert-bc-gripper-loss-weight 2.0 \
  --expert-bc-action-std 0.05 \
  --expert-dagger-iterations 3 \
  --expert-dagger-rollouts 3 \
  --expert-dagger-epochs 200 \
  --expert-dagger-max-steps 250 \
  --total-timesteps 2048 \
  --horizon 1000 \
  --learning-rate 0.0001 \
  --n-steps 256 \
  --batch-size 256 \
  --n-epochs 3 \
  --device cpu \
  --seed 55 \
  --output-dir outputs/ppo_original_handover_safe_ik_bc_smoke
```

短縮コマンド:

```bash
make ppo-original-handover-bc-smoke
```

学習済み policy の task reward 評価では、報酬 shaping を入れずに `safe_ik` stack で確認します。

```bash
MUJOCO_GL=egl python -m allostatic_handover.experiments.eval_ppo \
  --backend hrgym \
  --handover-env original \
  --hrgym-wrapper-stack safe_ik \
  --hrgym-shield-type PFL \
  --reward-variant task_only \
  --model-path outputs/ppo_original_handover_safe_ik_bc_smoke/model_final.zip \
  --episodes 3 \
  --horizon 1000 \
  --device cpu \
  --output-dir outputs/eval_ppo_original_handover_safe_ik_bc_smoke
```

短縮コマンド:

```bash
make eval-ppo-original-handover-bc-smoke
```

allostatic版の長めの実験:

```bash
python -m allostatic_handover.experiments.train_ppo \
  --backend hrgym \
  --handover-env allostatic \
  --reward-variant allostatic \
  --total-timesteps 200000 \
  --horizon 1000 \
  --output-dir outputs/ppo_hrgym_allostatic
```

## PPO 比較実験

要件定義の主比較は、同じ MuJoCo / human-robot-gym allostatic handover 環境を使い、reward variant だけを切り替えます。

- `task_only`
- `speech_penalty`
- `allostatic`

allostatic 環境では、人間側に hidden な `human_readiness` を持たせています。発話は `human_readiness` を上げ、`human_readiness >= threshold` になると MuJoCo の人間アニメーションが `REACH_OUT` に進み、手を伸ばし始めます。`ANNOUNCE_HANDOVER` や `SAY_RELEASING` のような意味の通る手渡し合図を出した後は、`readiness_hold_steps` の間 `human_readiness` を `readiness_hold_floor` 以上に保ちます。そのため、ロボットは発話を毎ステップ繰り返さなくても、人が準備した状態を一定時間利用して運搬できます。保持時間が切れた後に沈黙が続くと `human_readiness` は徐々に下がり、閾値を下回ると手伸ばし区間は再び readiness gate で止まります。手を伸ばしている状態を準備完了とみなし、接触成功には `human_reach_progress` が十分進んでいることも要求します。

発話は同時に allostatic load も上げます。load は通常の範囲では成功可否に直結させず、人間の適応能力として「load は上がるが受け取り自体は大きく崩れない」設計にしています。極端な withdrawal 状態だけは例外です。policy observation には真値ではなく `human_readiness_belief` と `allostatic_load_proxy` を入れ、`human_state` / `allostatic_load_total` の真値は `--privileged-observation` の時だけ入ります。

3条件を同じ seed / horizon / `safe_ik_air` / BC初期化条件で順に学習・評価する場合:

`safe_ik_air` は4次元 motor action にだけ action-based expert imitation reward を加えます。allostatic 版では5次元 action の最後の speech scalar を先に分離し、AIR は `(dx, dy, dz, gripper)` の安定化にだけ使います。3条件すべてで同じ wrapper を使うため、比較対象の差は speech/load 系の報酬項に残ります。

```bash
MUJOCO_GL=egl python -m allostatic_handover.experiments.train_ppo_comparison \
  --backend hrgym \
  --handover-env allostatic \
  --hrgym-wrapper-stack safe_ik_air \
  --hrgym-shield-type PFL \
  --total-timesteps 500000 \
  --eval-episodes 5 \
  --horizon 1000 \
  --seed 101 \
  --device cpu \
  --wandb-mode online \
  --wandb-project allostatic-handover-mvp \
  --wandb-group ppo_allostatic_readiness_hold_eta_air_compare \
  --expert-bc-speech-policy excessive_speech \
  --output-root outputs/ppo_allostatic_readiness_hold_eta_air_compare_seed101
```

tmuxで本格実験を開始:

```bash
make tmux-ppo-allostatic-compare-safe-ik
```

進捗確認:

```bash
./.conda/bin/tmux attach -t allostatic_ppo_readiness_hold_eta_air_compare
make tail-ppo-allostatic-compare-safe-ik
```

比較結果は `outputs/ppo_allostatic_readiness_hold_eta_air_compare_seed101/comparison_summary.csv` に保存されます。各variantの学習ログ、checkpoint、評価ログは同じディレクトリ配下に分かれて保存されます。

学習中の ETA は terminal / W&B / TensorBoard に記録されます。PPO rollout 中は `time/eta_seconds`, `time/eta_minutes`, `time/eta_hours`, `time/progress_percent`, `time/remaining_timesteps` が出力され、ローカルには各runの `eta.jsonl` にも保存されます。expert BC 中は epoch ログに `elapsed`, `eta`, `progress` が表示され、`expert_bc_eta.jsonl` と W&B の `bc/*` metrics にも保存されます。

`train_ppo_comparison` のBC初期化は、成功expert rolloutを優先して使います。比較runのデフォルト発話BCは `excessive_speech` で、3条件とも同じ高発話初期方策から始めます。`task_only` は発話/loadを下げる報酬圧がないため高発話を維持しやすく、`allostatic` は成功を維持しつつ発話/loadを下げる方向の更新が入る、という差を確認する設計です。成功サンプルが取れない場合だけ、allostatic FSMによりrunが止まらないよう失敗trajectoryへフォールバックします。

config YAMLを指定する場合:

```bash
MUJOCO_GL=egl python -m allostatic_handover.experiments.train_ppo \
  --backend hrgym \
  --handover-env allostatic \
  --hrgym-wrapper-stack safe_ik \
  --reward-config configs/reward_allostatic.yaml \
  --human-fsm-config configs/human_states.yaml \
  --reward-weight allostatic_load=0.10 \
  --human-fsm-param overload_threshold=3.5
```

## human-robot-gym Exact RHH-SAC

human-robot-gym の ICRA 2024 `RobotHumanHandoverCart` 用 `RHH-SAC.yaml` と `dataset_creation/RHH.yaml` の設定値で、dataset 作成から SAC 学習まで実行します。human-robot-gym 本体は変更せず、この repo 側の launcher が現在の Gymnasium / SB3 との互換差分だけを吸収します。

元設定の主な値:

- environment: `RobotHumanHandoverCart`, `shield_type=PFL`, `horizon=1000`, `control_freq=10`
- wrappers: collision prevention, IK position delta `action_limit=0.1`, action imitation reward `alpha=0.0`, dataset observation normalization
- algorithm: `SAC`, `learning_rate=0.0005`, `buffer_size=1000000`, `batch_size=128`, `train_freq=(100, step)`, `ent_coef=auto_0.2`
- run: dataset `robot-human-handover` を 500 episode 作成後、`n_envs=8`, `n_steps=3000001`

smoke:

```bash
MUJOCO_GL=egl python -m allostatic_handover.experiments.hrgym_exact_rhh_sac \
  --mode create-dataset-and-train \
  --dataset-name robot-human-handover-codex-smoke \
  --dataset-episodes 1 \
  --train-steps 128 \
  --n-envs 1 \
  --run-id codex_smoke_rhh_sac \
  --output-dir outputs/hrgym_exact_rhh_sac_smoke \
  --overwrite-dataset
```

本番:

```bash
MUJOCO_GL=egl python -m allostatic_handover.experiments.hrgym_exact_rhh_sac \
  --mode create-dataset-and-train \
  --output-dir outputs/hrgym_exact_rhh_sac_full
```

tmux で回す場合:

```bash
tmux new-session -d -s hrgym_exact_rhh_sac \
  "cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover && \
   MUJOCO_GL=egl ./.conda/bin/python -u -m allostatic_handover.experiments.hrgym_exact_rhh_sac \
     --mode create-dataset-and-train \
     --output-dir outputs/hrgym_exact_rhh_sac_full \
     > outputs/hrgym_exact_rhh_sac_full/run.log 2>&1"
```

進捗確認:

```bash
tmux ls
tail -f outputs/hrgym_exact_rhh_sac_full/run.log
find /mnt/k_iwamoto/sim_data/Projects/human-robot-gym/datasets/robot-human-handover \
  -maxdepth 1 -type d -name 'ep_*' | wc -l
```

## W&B

デフォルトは disabled です。offline で使う例:

```bash
python -m allostatic_handover.experiments.run_scripted_rollouts \
  --backend hrgym \
  --policy excessive_speech \
  --reward-variant task_only \
  --wandb-mode offline
```

API key はコードに書かず、通常の `wandb login` または環境変数で設定してください。

## GUI

学習・評価ログを見ます。環境の2D表示、ロボット/人の会話、hidden state と allostatic load の推移を確認できます。

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
python -m allostatic_handover.dashboard.app --log-dir outputs --port 7860
```

ブラウザで `http://127.0.0.1:7860` を開きます。

## MuJoCo Viewer

human-robot-gym の3D環境をその場で確認する場合は、Web GUIではなく robosuite / MuJoCo の `mjviewer` を使います。これはGUIデスクトップ上のローカルPCで実行してください。

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
conda activate /mnt/k_iwamoto/sim_data/Projects/allostatic-handover/.conda
MUJOCO_GL=glfw python -m allostatic_handover.experiments.run_scripted_rollouts \
  --backend hrgym \
  --policy minimal_speech \
  --reward-variant task_only \
  --episodes 1 \
  --horizon 1000 \
  --render \
  --print-step-info \
  --output-dir outputs/hrgym_mujoco_viewer
```

短縮コマンド:

```bash
make mujoco-viewer
```

`--render` は `has_renderer=True`, `renderer=mjviewer`, `render_camera=None` で human-robot-gym 環境を起動し、各stepで `env.render()` を呼びます。会話、human hidden state、allostatic load はターミナルにも出力され、同時に `outputs/hrgym_mujoco_viewer/steps.jsonl` に保存されます。

学習済み PPO policy を MuJoCo viewer で可視化評価する場合は、Linux のGUIデスクトップ上で次を実行します。

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
conda activate /mnt/k_iwamoto/sim_data/Projects/allostatic-handover/.conda
MUJOCO_GL=glfw python -m allostatic_handover.experiments.eval_ppo \
  --backend hrgym \
  --handover-env original \
  --hrgym-wrapper-stack safe_ik \
  --hrgym-shield-type PFL \
  --reward-variant task_only \
  --model-path outputs/ppo_original_handover_actor_critic_stable_20260611/best_model.zip \
  --episodes 3 \
  --horizon 1000 \
  --device cpu \
  --seed 12071 \
  --render \
  --print-step-info \
  --print-interval 25 \
  --output-dir outputs/eval_ppo_original_handover_actor_critic_stable_best_gui
```

短縮コマンド:

```bash
make eval-ppo-original-handover-stable-gui
```

`MUJOCO_GL=glfw` はローカル画面に viewer を出すための設定です。SSH先やDISPLAYのないheadless環境では viewer が開かないので、その場合はVNC/X11 forwarding/ローカルデスクトップから実行してください。

## Docker

`human-robot-gym` を BuildKit の named context として渡します。

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
DOCKER_BUILDKIT=1 docker build \
  --build-context hrgym=../human-robot-gym \
  -t allostatic-handover:dev .
docker run --rm -it --gpus all -p 7860:7860 \
  -v /mnt/k_iwamoto/sim_data/Projects/allostatic-handover/outputs:/workspace/allostatic-handover/outputs \
  allostatic-handover:dev
```

コンテナ内で実験を実行:

```bash
python -m allostatic_handover.experiments.run_scripted_rollouts --backend mock --episodes 2
python -m allostatic_handover.experiments.run_scripted_rollouts --backend hrgym --policy minimal_speech --episodes 1
```

## Logs

各 run directory に以下が出ます。

- `steps.jsonl`: step-level observation/action/info。GUIが読む。
- `episodes.jsonl`: episode-level metrics。
- `episodes.csv`: 解析用CSV。

主要 metric:

- `success`
- `return`
- `handover_time`
- `allostatic_load_total`
- `attention_load`
- `turn_taking_load`
- `proxemic_stress`
- `motor_adaptation_cost`
- `annoyance`
- `trust`
- `robot_speech_count`
- `human_waiting_time`
- `human_reach_effort`
- `human_readiness`
- `human_readiness_belief`
- `human_readiness_mean`
- `human_readiness_final`
- `readiness_blocked_count`
- `readiness_hold_steps_remaining`
- `human_reach_progress`
- `human_reach_progress_mean`
- `animation_gated_by_readiness`
- `reach_out_started_count`
- `allostatic_load_proxy`
- `withdrawal_count`
- `overload_count`

## Notes

- `human_state` と `allostatic_load` の真値は通常 observation には入れず、`info` とログに出します。
- `--privileged-observation` を付けた場合だけ observation に追加します。
- `backend=mock` は検証用です。論文・実験結果には `backend=hrgym` のログを使ってください。
- `safety_shield_py.py` は `shield_type=OFF` の smoke test 用 fallback です。SaRA safety shield を評価に使う場合は、`human-robot-gym/human_robot_gym/controllers/failsafe_controller/sara-shield` の CMake build を別途直して compiled binding を使ってください。
