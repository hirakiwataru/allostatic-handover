# allostatic-handover

`human-robot-gym` の手渡し課題を直接改造せず、外部リポジトリとして allostatic load、発話action、人間の隠れ状態FSM、Mjlab/MuJoCoタスク、PPO比較条件、DreamerV3系の状態推定・方策学習を実装するためのプロジェクトです。

現在の主実験は、Mjlab上の `Mjlab-Allostatic-Handover-Full-*` 系タスクで、YAMロボットが机上のハンマーを人の手に渡す課題です。ロボットは運動actionに加えて発話actionを選び、人側のreadinessやhidden FSMは環境内部で更新されます。

## Repository Layout

- `allostatic_handover/mjlab_tasks/`: Mjlab外部タスク、MDP term、報酬、観測、登録処理
- `allostatic_handover/world_model/`: PPO条件で使うPyTorch GRU belief estimator
- `allostatic_handover/dreamerv3_exact/`: JAX版DreamerV3本体を参照するbridgeとAllostatic Dreamer agent
- `allostatic_handover/envs/`: 旧human-robot-gym/mock backend用のallostatic load、FSM、speech token
- `allostatic_handover/dashboard/`: 可視化評価時の発話・FSM・load診断GUI
- `scripts/`: asset copy、Mjlab学習・評価、world model学習、DreamerV3学習
- `tests/`: unit testsと軽量visual check
- `ASSET_PROVENANCE.md`: HRGym由来assetのコピー元と扱い

生成物、学習ログ、vendor assetはgit管理しません。主な出力先は `outputs/`、HRGym assetコピー先は `assets/vendor/human_robot_gym/` です。

## Fresh Setup

このrepoは `human-robot-gym`、`mjlab`、必要に応じて `dreamerv3` を sibling repository として参照します。以下は `/mnt/k_iwamoto/sim_data/Projects` に配置する例です。

```bash
export PROJECTS=/mnt/k_iwamoto/sim_data/Projects
mkdir -p "$PROJECTS"
cd "$PROJECTS"

git clone https://github.com/mujocolab/mjlab.git
git clone https://github.com/TUMcps/human-robot-gym.git
git clone https://github.com/danijar/dreamerv3.git
git clone https://github.com/hirakiwataru/allostatic-handover.git
```

Mjlab側のPython環境を作ります。Mjlabの詳細な依存は上流READMEに従ってください。

```bash
cd "$PROJECTS/mjlab"
uv sync
```

このrepoをMjlab環境に外部task packageとしてeditable installします。

```bash
cd "$PROJECTS/allostatic-handover"
make mjlab-install
```

Full taskはHRGymのhuman XML、mesh、animation metadataをローカルコピーして使います。ライセンス確認前のvendor assetはgit管理対象外です。

```bash
cd "$PROJECTS/allostatic-handover"
python3 scripts/copy_hrgym_assets.py \
  --hrgym-root "$PROJECTS/human-robot-gym" \
  --include-full-handover-assets
```

登録確認:

```bash
cd "$PROJECTS/allostatic-handover"
make mjlab-list-envs
```

## Main Mjlab Tasks

| Task ID | Purpose |
|---|---|
| `Mjlab-Allostatic-Handover-Yam` | `Mjlab-Lift-Cube-Yam` ベースの簡易手渡しタスク |
| `Mjlab-Allostatic-Handover-Full` | YAM + HRGym風human/table/hammer配置のallostatic Full task |
| `Mjlab-Allostatic-Handover-Full-TaskOnlySpeech` | 発話あり、task rewardのみの比較条件 |
| `Mjlab-Allostatic-Handover-Full-SpeechPenalty` | 発話頻度で蓄積するhidden speech loadを罰する比較条件 |
| `Mjlab-Allostatic-Handover-Full-AllostaticBelief` | frozen belief estimatorを観測に加え、allostatic rewardで学習するPPO比較条件 |
| `Mjlab-Allostatic-Handover-Full-DreamerV3Allostatic` | PPOではなくJAX版DreamerV3 actor-criticで使うallostatic条件 |

Full系タスクの共通設定:

- Robot: YAM
- Object: hammer-like object
- Human/table/animation: HRGym `RobotHumanHandover` assetをコピーして使用
- Human animation: 現在の主比較では `RobotHumanHandover/0`
- Object pose: 固定机上配置 `(x=0.50, y=-0.16, z=0.88)`、yaw `pi/2`
- Action: `[dx, dy, dz, gripper, speech_scalar]`
- Speech token: `speech_scalar` を7種類の離散発話へ丸める

## Speech Tokens

`speech_scalar ∈ [-1, 1]` は以下のtokenへ変換されます。

```text
token = round((clip(speech_scalar, -1, 1) + 1) / 2 * 6)
```

| Token | Text |
|---|---|
| `SILENCE` |  |
| `ANNOUNCE_HANDOVER` | 今から渡します |
| `ASK_READY` | 準備できましたか |
| `REASSURE` | ゆっくりで大丈夫です |
| `SAY_WAITING` | 待ちます |
| `SAY_RELEASING` | 離します |
| `ASK_CONFIRMATION` | 取れましたか |

## Human Readiness, FSM, And Load

人の受け取り準備度 `readiness ∈ [0, 1]` は隠れ状態です。初期値は `0.24`、準備完了閾値は `0.62` です。`ANNOUNCE_HANDOVER`、`REASSURE`、`SAY_RELEASING` などの意味のある発話でreadinessが上がり、loadに応じて時間とともに下がります。

主要な発話効果:

```text
ANNOUNCE_HANDOVER: 0.52
ASK_READY:         0.18
REASSURE:          0.24
SAY_WAITING:       0.06
SAY_RELEASING:     0.34
ASK_CONFIRMATION:  0.04
```

同じ発話を連続した場合、readiness上昇効果は `0.25` 倍に弱まります。`ANNOUNCE_HANDOVER`、`REASSURE`、`SAY_RELEASING` は `180 step` のholdを発生させ、hold中のreadinessは最低 `0.78` を保ちます。

人の総合負荷 `allostatic_load_total` は以下です。

```text
allostatic_load_total =
  attention_load
+ turn_taking_load
+ proxemic_stress
+ motor_adaptation_cost
+ human_waiting_cost
+ human_reach_effort
```

各成分:

```text
attention_load =
  max(0, 0.975 * attention_load
         + 0.035 * 1[active_speech]
         + 0.055 * 1[token = ASK_READY]
         - 0.008)

turn_taking_load =
  max(0, 0.975 * turn_taking_load
         + 0.11 * 1[repeated_speech]
         - 0.008)

proxemic_stress =
  max(0, 0.975 * proxemic_stress
         + 0.65 * max(0, 0.18 - distance(robot_ee, human_hand))
         - 0.008)

human_reach_effort = 0.22 * reach_progress
human_waiting_cost = 0.08 * 1[phase = REACH_OUT] * clip(1 - reach_progress, 0, 1)
motor_adaptation_cost = 0.25 * |human_readiness - internal_readiness_belief|
```

FSM状態:

| State | Meaning |
|---|---|
| `READY` | 受け取り準備ができている |
| `HESITANT` | ためらっている |
| `DISTRACTED` | 注意が逸れている |
| `OVERLOADED` | 負荷が高い |
| `WITHDRAWING` | 受け取りから引いている |
| `GRASPING` | 受け取り動作中 |

FSM判定には総負荷からwithdrawal recoveryを引いた実効loadを使います。

```text
effective_load = allostatic_load_total - 4.0 * withdrawal_recovery

if reach_progress > 0.05: state = GRASPING
elif readiness >= 0.62:   state = READY
else:                     state = HESITANT

if effective_load >= 7.0: state = OVERLOADED
if effective_load >= 9.0: state = WITHDRAWING
```

`WITHDRAWING` 中でも `ANNOUNCE_HANDOVER` を出すと、`withdrawal_recovery` が緩やかに上がり、FSM判定用の実効loadだけを下げます。ログ・報酬用のload本体は消えません。

## PPO Comparison Conditions

3条件ともYAM、机、ハンマー、人アニメーション、action空間、task shapingは共通です。actorとcriticの両方から、真の `human_state_id`、`human_readiness`、`allostatic_load_total`、手書きの `readiness_belief`、`load_proxy` を外しています。

| Condition | Task ID | Observation | Reward |
|---|---|---|---|
| TaskOnlySpeech | `Mjlab-Allostatic-Handover-Full-TaskOnlySpeech` | 公開観測のみ | `R_task` |
| SpeechPenalty | `Mjlab-Allostatic-Handover-Full-SpeechPenalty` | TaskOnlySpeechと同じ | `R_task - 0.04 * speech_penalty` |
| AllostaticBelief | `Mjlab-Allostatic-Handover-Full-AllostaticBelief` | 公開観測 + frozen belief estimator出力 | `R_task - 0.02 * speech_penalty - 0.05 * allostatic_load_total - 0.10 * human_waiting_cost` |

共通task rewardの主要項:

```text
R_task =
  0.4  * handover_precision
+ 30.0 * success
+ 1.2  * robot_grasp_approach
+ 0.15 * robot_grasp
+ 0.25 * carry_to_hand
+ 5.0  * release_at_hand
+ 25.0 * handoff
- 0.02 * time_penalty
- 0.002 * action_rate_l2
+ joint_limit_regularization
```

発話ペナルティは、発話回数そのものではなく、発話で蓄積するhidden speech loadに対する指数型ペナルティです。

```text
speech_load = attention_load + turn_taking_load
excess = clip(speech_load - 0.8, 0, 4.0)
speech_penalty = expm1(0.5 * excess)
```

このため、少ない発話や間隔の空いた発話はほぼ罰せず、高頻度発話や同じ発話の繰り返しが続いた時だけ滑らかに強く罰します。

## Belief Estimator Used By AllostaticBelief

`AllostaticBelief` は、公開観測と5D action履歴から人の隠れ状態を推定する凍結PyTorch GRU belief estimatorを観測に追加します。真のFSM/readiness/loadはpolicyには渡しません。

公開観測:

```text
public_obs = [
  joint_pos,
  joint_vel,
  ee_to_object,
  ee_to_hand,
  object_to_hand,
  speech_context,
  phase_progress
]
```

GRU:

```text
x_t = normalize([public_obs_t, action_t])
h_t = GRUCell(x_t, h_{t-1})
belief_t = tanh(W_b LayerNorm(h_t))
human_state_probs_t = softmax(W_s h_t)
readiness_pred_t = sigmoid(W_r h_t)
load_pred_t = softplus(W_l h_t)
```

学習loss:

```text
Loss =
  CE(human_state_probs, human_state_id)
+ MSE(readiness_pred, human_readiness)
+ 0.2 * MSE(load_pred, allostatic_load_total)
```

PPO実行時には `belief_t`、`human_state_probs_t`、`readiness_pred_t`、`load_pred_t` だけがactor/critic観測に追加されます。

## Train And Evaluate The Three PPO Conditions

3条件を1つのtmux session内で順番に実行します。順序は `TaskOnlySpeech -> SpeechPenalty -> world-model dataset収集 -> belief world model学習 -> AllostaticBelief PPO` です。

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
NUM_ENVS=64 MAX_ITER=500 WM_STEPS=4096 WM_UPDATES=5000 \
  make tmux-mjlab-train-three-condition-sequence-wandb-gpu
make tail-mjlab-three-condition-sequence
```

短いsmoke:

```bash
NUM_ENVS=8 MAX_ITER=2 WM_STEPS=128 WM_UPDATES=10 SLEEP_SECONDS=2 \
  make mjlab-train-three-condition-sequence-wandb-gpu
```

評価:

```bash
make mjlab-eval-full-task-only-speech CKPT=/path/to/task_only_speech/model.pt
make mjlab-eval-full-speech-penalty CKPT=/path/to/speech_penalty/model.pt
make mjlab-eval-full-allostatic-belief \
  CKPT=/path/to/allostatic_belief/model.pt \
  WM_MODEL=/path/to/belief_distill.pt
```

2000 iterationまで追加学習したseed 101のheadless評価では、3条件すべてが64 episodeで成功率1.0に到達しました。差が出たのは主に発話とloadです。

| Condition | Success | Speech Count | Repeated Speech | Silence Ratio | Load Mean | Mean Reward |
|---|---:|---:|---:|---:|---:|---:|
| TaskOnlySpeech | 1.00 | 586.5 | 488.2 | 0.276 | 2.987 | 14927.3 |
| SpeechPenalty | 1.00 | 150.2 | 66.7 | 0.815 | 0.438 | 14942.1 |
| AllostaticBelief | 1.00 | 126.1 | 56.3 | 0.845 | 0.425 | 14941.2 |

この結果は、AllostaticBeliefが「成功率を維持しつつ発話・負荷を最も抑える」方向に働いたことを示します。ただし、AllostaticBeliefではbelief観測とload/waiting報酬が同時に追加されているため、belief単独の寄与を主張するには追加アブレーションが必要です。

## Visualization

Linuxモニタでnative MuJoCo viewerを開く場合:

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
make mjlab-install
make mjlab-check-display
make mjlab-play-full-task-only-speech
```

このLinux PCではデフォルトを `DISPLAY=:1`、`XAUTHORITY=/run/user/<uid>/gdm/Xauthority` にしています。別環境では上書きしてください。

```bash
make mjlab-play-full-task-only-speech \
  MJLAB_DISPLAY=:0 \
  MJLAB_XAUTHORITY="$XAUTHORITY"
```

学習済みpolicyを発話/FSM/loadのライブGUI付きで確認:

```bash
make mjlab-play-full-task-only-speech-trained-live CKPT=/path/to/model.pt
```

別ブラウザで `http://127.0.0.1:7860/live.html` を開くと、発話履歴、人のhidden state、readiness、load、palm distance、handoff gateを確認できます。ログは `outputs/mjlab_live/*/live.jsonl` に保存されます。

## Exact DreamerV3 Actor-Critic

PPO 3条件とは別に、`/mnt/k_iwamoto/sim_data/Projects/dreamerv3` のJAX版DreamerV3本体を使ってオンラインactor-critic学習を回す導線があります。対象taskは `Mjlab-Allostatic-Handover-Full-DreamerV3Allostatic` です。

依存環境:

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
make dreamerv3-exact-mjlab-venv
make dreamerv3-exact-check-deps
make dreamerv3-exact-check-jax-cpu
```

GPU確認:

```bash
make dreamerv3-exact-check-jax-gpu
```

JAX GPUはPyTorch/MjlabのCUDAとは別です。`Unknown backend cuda` や `DNN library initialization failed` が出る場合は、`.dreamerv3-venv` 内のJAX CUDA/cuDNN wheelとローカルNVIDIA driverの整合が取れていません。その場合でもCPU smokeで実装確認はできます。

Smoke:

```bash
make dreamerv3-mjlab-policy-smoke
```

本格run:

```bash
make tmux-dreamerv3-mjlab-policy-wandb
make tail-dreamerv3-mjlab-policy
```

評価:

```bash
make dreamerv3-mjlab-policy-eval \
  DREAMERV3_POLICY_CKPT=/path/to/policy.ckpt
```

DreamerV3 policy入力は `public_obs` のみです。真の `human_state_id`、`human_readiness`、`allostatic_load_total` はreplayの補助教師ラベルとして保存しますが、policy observationには入れません。

## Tests

単体テスト:

```bash
cd /mnt/k_iwamoto/sim_data/Projects/allostatic-handover
python3 -m unittest discover -s tests
```

Mjlab Full visual checkを含める場合は、HRGym assetをコピーして `MUJOCO_GL=egl` を指定します。

```bash
make copy-hrgym-full-assets
MUJOCO_GL=egl python3 -m unittest discover -s tests
```

`tests/test_zz_mjlab_full_visual_check.py` は `outputs/visual_checks/mjlab_full/` に `reset.png`、`reach_out.png`、`handoff.png` を保存します。

## Legacy human-robot-gym Backend

`allostatic_handover/experiments/` には、`human-robot-gym` の `RobotHumanHandoverCart` / `AllostaticRobotHumanHandoverCart` をGymnasium adapter経由でStable-Baselines3 PPOに渡す旧導線も残しています。

```bash
conda env create -f environment.yml
conda activate allostatic-handover

MUJOCO_GL=egl python -m allostatic_handover.experiments.train_ppo \
  --backend hrgym \
  --handover-env allostatic \
  --hrgym-wrapper-stack safe_ik \
  --reward-variant task_only \
  --total-timesteps 128 \
  --horizon 20 \
  --device cpu \
  --output-dir outputs/ppo_allostatic_safe_ik_smoke
```

現在の主比較はMjlab側です。旧backendは、HRGym実装との差分確認やsafe-IK stackの診断用として維持しています。

## Notes On Git And Assets

- `outputs/`, `.uvcache/`, `.dreamerv3-venv/`, `assets/vendor/` はgitに含めません。
- HRGym由来assetはローカルコピーで使用します。配布可否は上流ライセンス確認が必要です。
- W&Bログは `allostatic-handover-mjlab` projectへ送る想定です。
- このrepoは `mjlab`、`human-robot-gym`、`dreamerv3` 本体を変更しません。
