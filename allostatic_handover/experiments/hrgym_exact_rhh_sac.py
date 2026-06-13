"""Run the original human-robot-gym RHH-SAC setup from this repository.

The launcher keeps the configuration values from human-robot-gym's
``RHH-SAC.yaml`` and adds only compatibility glue for the currently installed
Gymnasium / Stable-Baselines3 stack. It does not modify human-robot-gym files.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from allostatic_handover.wrappers.hrgym_training_stack import (
    EXPERT_HANDOVER_OBS_KEYS,
    _ActionBasedExpertRewardCompatWrapper,
    _ExpertObsCompatWrapper,
    _ScalarPickPlaceHumanCartExpert,
)
from allostatic_handover.wrappers.robosuite_gymnasium import ORIGINAL_HANDOVER_OBS_KEYS, adapt_robosuite_for_sb3


DEFAULT_HRGYM_ROOT = Path("/mnt/k_iwamoto/sim_data/Projects/human-robot-gym")
DEFAULT_TRAIN_CONFIG = (
    DEFAULT_HRGYM_ROOT
    / "human_robot_gym/training/config_icra_2024/environment_evaluation/training/RHH-SAC.yaml"
)
DEFAULT_DATASET_CONFIG = (
    DEFAULT_HRGYM_ROOT
    / "human_robot_gym/training/config_icra_2024/environment_evaluation/dataset_creation/RHH.yaml"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["create-dataset", "train", "create-dataset-and-train"],
        default="create-dataset-and-train",
    )
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-episodes", type=int, default=None)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/hrgym_exact_rhh_sac"))
    parser.add_argument("--overwrite-dataset", action="store_true")
    parser.add_argument("--skip-dataset-if-present", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _apply_runtime_compatibility_patches()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / ".mplconfig"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    train_cfg = _load_config(args.train_config)
    dataset_cfg = _load_config(args.dataset_config)
    _apply_cli_overrides(train_cfg=train_cfg, dataset_cfg=dataset_cfg, args=args)

    _save_resolved_config(train_cfg, args.output_dir / "resolved_RHH-SAC.yaml")
    _save_resolved_config(dataset_cfg, args.output_dir / "resolved_RHH_dataset.yaml")

    if args.mode in {"create-dataset", "create-dataset-and-train"}:
        dataset_dir = _dataset_dir(dataset_cfg.dataset_name)
        if _dataset_ready(dataset_dir) and args.skip_dataset_if_present and not args.overwrite_dataset:
            print(f"dataset already present: {dataset_dir}")
        else:
            if dataset_dir.exists() and not args.overwrite_dataset:
                raise FileExistsError(
                    f"Dataset directory exists but is incomplete: {dataset_dir}. "
                    "Pass --overwrite-dataset to regenerate it."
                )
            _create_dataset(config=dataset_cfg, output_dir=args.output_dir, overwrite=args.overwrite_dataset)

    if args.mode in {"train", "create-dataset-and-train"}:
        _train_exact_sac(config=train_cfg, output_dir=args.output_dir)

    return 0


def _load_config(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return OmegaConf.load(path)


def _apply_cli_overrides(train_cfg, dataset_cfg, args: argparse.Namespace) -> None:
    if args.dataset_name is not None:
        train_cfg.run.dataset_name = args.dataset_name
        train_cfg.wrappers.action_based_expert_imitation_reward.dataset_name = args.dataset_name
        train_cfg.wrappers.dataset_obs_norm.dataset_name = args.dataset_name
        dataset_cfg.dataset_name = args.dataset_name
        dataset_cfg.run.dataset_name = args.dataset_name

    if args.dataset_episodes is not None:
        dataset_cfg.n_episodes = args.dataset_episodes
    if args.train_steps is not None:
        train_cfg.run.n_steps = args.train_steps
    if args.n_envs is not None:
        train_cfg.run.n_envs = args.n_envs
    if args.seed is not None:
        train_cfg.run.seed = args.seed
        train_cfg.environment.seed = args.seed
        train_cfg.algorithm.seed = args.seed
        train_cfg.expert.seed = args.seed
        dataset_cfg.run.seed = args.seed
        dataset_cfg.environment.seed = args.seed
        dataset_cfg.algorithm.seed = args.seed
        dataset_cfg.expert.seed = args.seed
    if args.run_id is not None:
        train_cfg.run.id = args.run_id
        train_cfg.wandb_run.name = args.run_id


def _save_resolved_config(config, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(OmegaConf.to_yaml(config, resolve=True), encoding="utf-8")


def _apply_runtime_compatibility_patches() -> None:
    """Allow current Gymnasium wrappers to wrap robosuite-style wrappers."""
    import gymnasium

    def permissive_wrapper_init(self, env):
        self.env = env
        self._action_space = None
        self._observation_space = None
        self._metadata = None
        self._cached_spec = None

    gymnasium.core.Wrapper.__init__ = permissive_wrapper_init


def _dataset_dir(dataset_name: str) -> Path:
    from human_robot_gym.utils.mjcf_utils import file_path_completion

    return Path(file_path_completion(f"../datasets/{dataset_name}"))


def _dataset_ready(dataset_dir: Path) -> bool:
    return (
        dataset_dir.exists()
        and (dataset_dir / "observations.csv").exists()
        and any(dataset_dir.glob("ep_*/state.npz"))
    )


def _create_dataset(config, output_dir: Path, overwrite: bool = False) -> None:
    import shutil

    dataset_dir = _dataset_dir(config.dataset_name)
    if dataset_dir.exists() and overwrite:
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    _save_resolved_config(config, dataset_dir / "config.yaml")

    env = _make_exact_env(config=config, dataset_mode=True)
    expert = _make_exact_expert(config=config, env=env)
    all_observations: list[np.ndarray] = []
    episode_lengths: list[int] = []
    episode_returns: list[float] = []
    successes = 0

    try:
        for episode_id in range(int(config.n_episodes)):
            obs = env.reset()
            raw_env = env.unwrapped
            model_xml = raw_env.sim.model.get_xml()
            state0 = raw_env.get_environment_state()
            raw_env.reset_from_xml_string(model_xml)
            raw_env.set_environment_state(state0)

            states = [state0]
            observations = [np.asarray(obs, dtype=np.float32)]
            expert_observations = [_current_expert_obs(env)]
            actions = []
            total_return = 0.0
            last_info: dict[str, Any] = {}

            for step_id in range(int(config.environment.horizon)):
                action = expert(_current_expert_obs(env))
                obs, reward, terminated, truncated, info = env.step(action)
                actions.append(np.asarray(action, dtype=np.float32))
                states.append(raw_env.get_environment_state())
                observations.append(np.asarray(obs, dtype=np.float32))
                expert_observations.append(_current_expert_obs(env))
                total_return += float(reward)
                last_info = dict(info)
                if terminated or truncated:
                    break

            ep_dir = dataset_dir / f"ep_{episode_id:06d}"
            ep_dir.mkdir(parents=False, exist_ok=False)
            (ep_dir / "model.xml").write_text(model_xml, encoding="utf-8")
            np.savez(
                ep_dir / "state.npz",
                states=np.asarray(states, dtype=object),
                observations=np.asarray(observations, dtype=np.float32),
                expert_observations=np.asarray(expert_observations, dtype=object),
                actions=np.asarray(actions, dtype=np.float32),
            )

            all_observations.extend(observations)
            episode_lengths.append(step_id + 1)
            episode_returns.append(total_return)
            if last_info.get("n_goal_reached", 0) > 0:
                successes += 1

            if (episode_id + 1) % 10 == 0 or episode_id == 0:
                print(
                    f"dataset episode {episode_id + 1}/{config.n_episodes}: "
                    f"len={step_id + 1} return={total_return:.3f} "
                    f"successes={successes}"
                )
    finally:
        env.close()

    _write_observation_stats(dataset_dir / "observations.csv", np.asarray(all_observations, dtype=np.float32))
    _write_dataset_stats(
        dataset_dir / "stats.csv",
        successes=successes,
        episode_lengths=episode_lengths,
        episode_returns=episode_returns,
    )
    print(f"wrote dataset to {dataset_dir}")


def _make_exact_env(config, dataset_mode: bool = False):
    import robosuite
    from human_robot_gym.utils.mjcf_utils import file_path_completion
    from human_robot_gym.utils.training_utils import _compose_environment_kwargs
    from human_robot_gym.wrappers.collision_prevention_wrapper import CollisionPreventionWrapper
    from human_robot_gym.wrappers.ik_position_delta_wrapper import IKPositionDeltaWrapper

    env_kwargs = _compose_environment_kwargs(config=config, evaluation_mode=False)
    raw_env = robosuite.make(config.environment.env_id, **env_kwargs)
    env = _ExpertObsCompatWrapper(
        raw_env,
        agent_keys=tuple(config.run.obs_keys or ORIGINAL_HANDOVER_OBS_KEYS),
        expert_keys=tuple(config.run.expert_obs_keys or EXPERT_HANDOVER_OBS_KEYS),
    )
    env = CollisionPreventionWrapper(
        env=env,
        collision_check_fn=raw_env.check_collision_action,
        replace_type=int(config.wrappers.collision_prevention.replace_type),
        n_resamples=int(config.wrappers.collision_prevention.n_resamples),
    )
    action_limit = float(config.wrappers.ik_position_delta.action_limit)
    env = IKPositionDeltaWrapper(
        env=env,
        urdf_file=file_path_completion(config.wrappers.ik_position_delta.urdf_file),
        action_limits=np.array(
            [[-action_limit, -action_limit, -action_limit], [action_limit, action_limit, action_limit]],
            dtype=np.float32,
        ),
        x_output_max=float(config.wrappers.ik_position_delta.x_output_max),
        x_position_limits=config.wrappers.ik_position_delta.x_position_limits,
        residual_threshold=float(config.wrappers.ik_position_delta.residual_threshold),
        max_iter=int(config.wrappers.ik_position_delta.max_iter),
    )
    _set_action_space_from_spec(env)

    if dataset_mode:
        return env

    env = _DatasetRSICompatWrapper(
        env=env,
        dataset_name=config.wrappers.action_based_expert_imitation_reward.dataset_name,
        rsi_prob=float(config.wrappers.action_based_expert_imitation_reward.rsi_prob),
    )
    _set_action_space_from_spec(env)
    expert = _make_exact_expert(config=config, env=env)
    env = _ActionBasedExpertRewardCompatWrapper(
        env=env,
        expert=expert,
        alpha=float(config.wrappers.action_based_expert_imitation_reward.alpha),
        beta=float(config.wrappers.action_based_expert_imitation_reward.beta),
        iota_m=float(config.wrappers.action_based_expert_imitation_reward.iota_m),
        iota_g=float(config.wrappers.action_based_expert_imitation_reward.iota_g),
        m_sim_fn=str(config.wrappers.action_based_expert_imitation_reward.m_sim_fn),
        g_sim_fn=str(config.wrappers.action_based_expert_imitation_reward.g_sim_fn),
    )
    env = _DatasetObsNormCompatWrapper(
        env=env,
        dataset_name=config.wrappers.dataset_obs_norm.dataset_name,
        squash_factor=config.wrappers.dataset_obs_norm.squash_factor,
    )
    return adapt_robosuite_for_sb3(env, obs_keys=None, append_speech_action=False, force_adapter=True)


def _make_exact_expert(config, env):
    expert_kwargs = {
        "signal_to_noise_ratio": float(config.expert.signal_to_noise_ratio),
        "hover_dist": float(config.expert.hover_dist),
        "tan_theta": float(config.expert.tan_theta),
        "horizontal_epsilon": float(config.expert.horizontal_epsilon),
        "vertical_epsilon": float(config.expert.vertical_epsilon),
        "goal_dist": float(config.expert.goal_dist),
        "gripper_fully_opened_threshold": float(config.expert.gripper_fully_opened_threshold),
        "release_when_delivered": bool(config.expert.release_when_delivered),
        "delta_time": float(config.expert.delta_time),
        "seed": None if config.expert.seed is None else int(config.expert.seed),
    }
    return _ScalarPickPlaceHumanCartExpert(
        observation_space=env.observation_space,
        action_space=env.action_space,
        **expert_kwargs,
    )


class _DatasetRSICompatWrapper:
    def __init__(self, env, dataset_name: str, rsi_prob: float = 0.0):
        self.env = env
        self.dataset = _load_dataset(dataset_name)
        self._rsi_prob = float(rsi_prob)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._dataset_transition_count = None
        self._dataset_ep_step_idx = None
        self._dic = None

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def action_spec(self):
        return self.env.action_spec

    def reset(self):
        self.env.reset()
        ep_idx = np.random.randint(len(self.dataset))
        xml, self._dic = self.dataset[ep_idx]
        self._dataset_transition_count = len(self._dic["states"]) - 1
        if np.random.rand() < self._rsi_prob:
            self._dataset_ep_step_idx = np.random.randint(self._dataset_transition_count)
        else:
            self._dataset_ep_step_idx = 0

        self.unwrapped.reset_from_xml_string(xml)
        self.unwrapped.set_environment_state(self._dic["states"][self._dataset_ep_step_idx])
        expert_wrapper = _find_expert_obs_wrapper(self.env)
        if expert_wrapper is not None:
            expert_wrapper._current_expert_observation = self._dic["expert_observations"][self._dataset_ep_step_idx]
            expert_wrapper._previous_expert_observation = None
        return np.asarray(self._dic["observations"][self._dataset_ep_step_idx], dtype=np.float32)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._dataset_ep_step_idx = min(self._dataset_ep_step_idx + 1, self._dataset_transition_count)
        return obs, reward, terminated, truncated, info

    def close(self):
        return self.env.close()

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def __getattr__(self, attr: str):
        return getattr(self.env, attr)


class _DatasetObsNormCompatWrapper:
    def __init__(self, env, dataset_name: str, squash_factor: float | None = None):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._squash_factor = squash_factor
        stats_path = _dataset_dir(dataset_name) / "observations.csv"
        if not stats_path.exists():
            raise FileNotFoundError(stats_path)
        rows = list(csv.DictReader(stats_path.open(encoding="utf-8")))
        self._obs_mean = np.asarray([float(row["mean"]) for row in rows], dtype=np.float32)
        self._obs_std = np.asarray([float(row["std"]) for row in rows], dtype=np.float32)
        self._obs_std[self._obs_std == 0] = 1.0

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def action_spec(self):
        return self.env.action_spec

    def reset(self):
        return self._normalize(self.env.reset())

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._normalize(obs), reward, terminated, truncated, info

    def close(self):
        return self.env.close()

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def __getattr__(self, attr: str):
        return getattr(self.env, attr)

    def _normalize(self, obs):
        obs = (np.asarray(obs, dtype=np.float32) - self._obs_mean) / self._obs_std
        if self._squash_factor is not None:
            obs = np.tanh(float(self._squash_factor) * obs)
        return obs.astype(np.float32, copy=False)


def _load_dataset(dataset_name: str):
    dataset_dir = _dataset_dir(dataset_name)
    if not dataset_dir.exists():
        raise FileNotFoundError(dataset_dir)
    episodes = []
    for ep_dir in sorted(dataset_dir.glob("ep_*")):
        xml_path = ep_dir / "model.xml"
        state_path = ep_dir / "state.npz"
        if not xml_path.exists() or not state_path.exists():
            continue
        with np.load(state_path, allow_pickle=True) as data:
            episodes.append(
                (
                    xml_path.read_text(encoding="utf-8"),
                    {
                        "states": data["states"],
                        "observations": data["observations"],
                        "expert_observations": data["expert_observations"],
                        "actions": data["actions"],
                    },
                )
            )
    if not episodes:
        raise FileNotFoundError(f"No ep_*/state.npz episodes found in {dataset_dir}")
    return episodes


def _train_exact_sac(config, output_dir: Path) -> None:
    from copy import deepcopy

    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    from human_robot_gym.callbacks.logging_callback import LoggingCallback

    n_envs = int(config.run.n_envs)
    seed = None if config.run.seed is None else int(config.run.seed)
    env_fns = [_make_env_fn(config, None if seed is None else seed + idx) for idx in range(n_envs)]
    vec_env_cls = DummyVecEnv if n_envs == 1 else SubprocVecEnv
    env = vec_env_cls(env_fns)

    algorithm_kwargs = OmegaConf.to_container(cfg=deepcopy(config.algorithm), resolve=True, throw_on_missing=True)
    algorithm_kwargs.pop("name")
    algorithm_kwargs.pop("create_eval_env", None)
    if isinstance(algorithm_kwargs.get("train_freq"), list):
        algorithm_kwargs["train_freq"] = tuple(algorithm_kwargs["train_freq"])
    algorithm_kwargs["env"] = env
    algorithm_kwargs["tensorboard_log"] = str(Path("runs") / config.wandb_run.project / config.wandb_run.group / config.wandb_run.name)

    model_path = Path("models") / config.wandb_run.project / config.wandb_run.group / config.wandb_run.name
    model_path.mkdir(parents=True, exist_ok=True)
    _save_resolved_config(config, model_path / "config.yaml")

    callback = LoggingCallback(
        verbose=2,
        save_freq=int(config.run.save_freq),
        model_path=str(model_path),
        start_episode=0,
        additional_log_info_keys=list(config.run.log_info_keys),
        log_interval=config.run.log_interval,
    )

    try:
        model = SAC(**algorithm_kwargs)
        model.learn(total_timesteps=int(config.run.n_steps), log_interval=None, callback=callback)
        model.save(model_path / "model_final")
        if hasattr(model, "save_replay_buffer"):
            model.save_replay_buffer(str(model_path / "replay_buffer"))
    finally:
        env.close()
    print(f"saved exact RHH-SAC model to {model_path.resolve()}")


def _make_env_fn(config, seed: int | None):
    cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))

    def _init():
        if seed is not None:
            cfg.environment.seed = seed
        return _make_exact_env(config=cfg, dataset_mode=False)

    return _init


def _set_action_space_from_spec(env) -> None:
    from gymnasium import spaces

    low, high = env.action_spec
    env.action_space = spaces.Box(
        low=np.asarray(low, dtype=np.float32),
        high=np.asarray(high, dtype=np.float32),
        dtype=np.float32,
    )


def _find_expert_obs_wrapper(env):
    while env is not None:
        if isinstance(env, _ExpertObsCompatWrapper):
            return env
        env = getattr(env, "env", None)
    return None


def _current_expert_obs(env) -> dict[str, Any]:
    wrapper = _find_expert_obs_wrapper(env)
    if wrapper is None:
        raise RuntimeError("Expert observation wrapper not found")
    return dict(wrapper._current_expert_observation)


def _write_observation_stats(path: Path, observations: np.ndarray) -> None:
    std = np.std(observations, axis=0)
    std[std == 0] = 1.0
    mean = np.mean(observations, axis=0)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["mean", "std"])
        writer.writeheader()
        for mean_i, std_i in zip(mean, std):
            writer.writerow({"mean": float(mean_i), "std": float(std_i)})


def _write_dataset_stats(
    path: Path,
    successes: int,
    episode_lengths: list[int],
    episode_returns: list[float],
) -> None:
    n_episodes = max(len(episode_lengths), 1)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["success_mean", "ep_len_mean", "ep_rew_mean"])
        writer.writeheader()
        writer.writerow(
            {
                "success_mean": successes / n_episodes,
                "ep_len_mean": float(np.mean(episode_lengths)) if episode_lengths else np.nan,
                "ep_rew_mean": float(np.mean(episode_returns)) if episode_returns else np.nan,
            }
        )


if __name__ == "__main__":
    raise SystemExit(main())
