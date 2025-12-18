# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Run a trained RL-Games PPO policy and record transitions to a pickle file.

Example (matches training config used for Factory PegInsert):
./isaaclab.sh -p custom/collect_rlgames_demos.py \
    --task Isaac-Factory-PegInsert-Direct-v0 \
    --num_envs 32 \
    --checkpoint ./logs/rl_games/Factory/test/nn/last_Factory_ep_200_rew_365.6033.pth \
    --num_steps 2000
"""

from __future__ import annotations

import argparse
import datetime
import os
import random
import sys
import time
import pickle as pkl
from typing import Any

import numpy as np
import torch

from isaaclab.app import AppLauncher

# CLI parsing (AppLauncher args come from Hydra defaults)
parser = argparse.ArgumentParser(description="Collect demos using an RL-Games PPO checkpoint.")
parser.add_argument("--task", type=str, default="Isaac-Factory-PegInsert-Direct-v0")
parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=32, help="Number of vectorized envs to roll out.")
parser.add_argument("--num_steps", type=int, default=200000, help="Max environment steps to record.")
parser.add_argument(
    "--trajectories",
    type=int,
    default=None,
    help="Total number of trajectories (episodes) to collect across all envs; stop after reaching this count.",
)
parser.add_argument(
    "--num_episodes",
    type=int,
    default=None,
    help="Deprecated alias for --trajectories (kept for backward compatibility).",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="./logs/rl_games/Factory/test/nn/last_Factory_ep_200_rew_365.6033.pth",
    help="Path to RL-Games checkpoint.",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="If checkpoint not provided, pick the last saved model under the run directory.",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment and agent.")
parser.add_argument("--real-time", action="store_true", default=False, help="Sleep to match sim dt.")
parser.add_argument(
    "--output",
    type=str,
    default="./datasets/ppo_factory_demos.pkl",
    help="Output pickle (timestamped suffix is added automatically).",
)

# Append simulator args
AppLauncher.add_app_launcher_args(parser)
# Parse known args; pass the rest to Hydra
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

# Launch simulator first
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# === Imports that require the simulator to be initialized ===
import gymnasium as gym
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


class RecordingRlGamesVecEnvWrapper(RlGamesVecEnvWrapper):
    """RlGames wrapper that also keeps the raw observations and actions for logging."""

    def reset(self):
        obs_dict, _ = self.env.reset()
        self.last_raw_obs = obs_dict
        self.last_actions = None
        self.last_terminated = None
        self.last_time_outs = None
        return self._process_obs(obs_dict)

    def step(self, actions):
        # Move to sim device and clip (mirrors base class)
        actions = actions.detach().clone().to(device=self._sim_device)
        actions = torch.clamp(actions, -self._clip_actions, self._clip_actions)
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)

        # Cache raw tensors for logging
        self.last_raw_obs = obs_dict
        self.last_actions = actions
        self.last_terminated = terminated
        self.last_time_outs = truncated

        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated.to(device=self._rl_device)

        obs_and_states = self._process_obs(obs_dict)
        rew = rew.to(device=self._rl_device)
        dones = (terminated | truncated).to(device=self._rl_device)
        extras = {
            k: v.to(device=self._rl_device, non_blocking=True) if hasattr(v, "to") else v for k, v in extras.items()
        }
        if "log" in extras:
            extras["episode"] = extras.pop("log")
        return obs_and_states, rew, dones, extras


def _tree_index_to_numpy(data: Any, idx: int) -> Any:
    """Slice per-env data and move to numpy on CPU."""
    if isinstance(data, dict):
        return {k: _tree_index_to_numpy(v, idx) for k, v in data.items()}
    if isinstance(data, torch.Tensor):
        arr = data.detach().cpu().numpy()
        return arr[idx].copy() if arr.ndim > 0 else arr.copy()
    if isinstance(data, np.ndarray):
        return data[idx].copy() if data.ndim > 0 else data.copy()
    return data


def _flatten_obs_to_vector(obs: Any) -> np.ndarray:
    """Flatten observation dict/tensor into a 1D float32 vector."""
    parts: list[np.ndarray] = []
    if isinstance(obs, dict):
        for v in obs.values():
            parts.append(_flatten_obs_to_vector(v))
    elif isinstance(obs, torch.Tensor):
        parts.append(obs.detach().cpu().numpy().reshape(-1))
    elif isinstance(obs, np.ndarray):
        parts.append(obs.reshape(-1))
    else:
        parts.append(np.array([obs], dtype=np.float32).reshape(-1))
    if len(parts) == 1:
        return parts[0].astype(np.float32)
    return np.concatenate(parts, axis=0).astype(np.float32)


def _get_success_flags(env) -> np.ndarray:
    """Fetch per-env success flags if available on the unwrapped env."""
    if hasattr(env.unwrapped, "ep_succeeded"):
        buf = getattr(env.unwrapped, "ep_succeeded")
        if isinstance(buf, torch.Tensor):
            return buf.detach().cpu().numpy().astype(bool)
        if isinstance(buf, np.ndarray):
            return buf.astype(bool)
        try:
            return np.array(buf, dtype=bool)
        except Exception:
            pass
    # Fallback: no success info available
    return None


def _resolve_checkpoint(agent_cfg: dict, task_name: str) -> str:
    """Match rl_games/play.py logic to find a checkpoint."""
    log_root_path = os.path.abspath(os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"]))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.checkpoint and os.path.exists(args_cli.checkpoint):
        return retrieve_file_path(args_cli.checkpoint)

    if args_cli.checkpoint and not os.path.exists(args_cli.checkpoint):
        print(f"[WARN] Provided checkpoint not found: {args_cli.checkpoint}")

    if args_cli.use_last_checkpoint:
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        checkpoint_file = ".*"
        return get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])

    # Try published checkpoints if available
    resume_path = get_published_pretrained_checkpoint("rl_games", task_name)
    if resume_path:
        return resume_path

    # Fallback to best checkpoint from training run
    run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
    checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
    return get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])


def _timestamped_path(base_path: str) -> str:
    """Add timestamp before extension."""
    base_path = os.path.abspath(base_path)
    folder, fname = os.path.split(base_path)
    stem, ext = os.path.splitext(fname)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{stem}_{ts}{ext}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Roll out checkpoint and record transitions."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # Override configs from CLI
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.device is not None:
        agent_cfg["params"]["config"]["device"] = args_cli.device
        agent_cfg["params"]["config"]["device_name"] = args_cli.device

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    env_cfg.seed = agent_cfg["params"]["seed"]

    # Find checkpoint
    resume_path = _resolve_checkpoint(agent_cfg, train_task_name)
    log_dir = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir

    # Create env
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Wrap for rl-games and register env creator
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", float("inf"))
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", float("inf"))
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)
    env = RecordingRlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # Load agent
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    dt = env.unwrapped.step_dt

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs.get("obs", obs)
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    transitions = []
    total_steps = 0
    episodes = 0
    target_steps = args_cli.num_steps
    target_episodes = args_cli.trajectories if args_cli.trajectories is not None else args_cli.num_episodes
    out_path = _timestamped_path(args_cli.output)

    while simulation_app.is_running():
        if target_steps is not None and total_steps >= target_steps:
            break
        if target_episodes is not None and episodes >= target_episodes:
            break

        start_time = time.time()
        with torch.inference_mode():
            obs_tensor = agent.obs_to_torch(obs)
            actions = agent.get_action(obs_tensor, is_deterministic=agent.is_deterministic)

            prev_raw_obs = env.last_raw_obs
            obs, rewards, dones, extras = env.step(actions)
            terminated = env.last_terminated
            truncated = env.last_time_outs
            actions_applied = env.last_actions

            if len(dones) > 0 and agent.is_rnn and agent.states is not None:
                for s in agent.states:
                    s[:, dones, :] = 0.0

        # Convert to numpy and record per-env transitions
        rewards_np = rewards.detach().cpu().numpy()
        dones_np = dones.detach().cpu().numpy().astype(bool)
        terminated_np = terminated.detach().cpu().numpy().astype(bool) if terminated is not None else dones_np
        truncated_np = truncated.detach().cpu().numpy().astype(bool) if truncated is not None else np.zeros_like(dones_np)
        actions_np = actions_applied.detach().cpu().numpy()
        success_flags = _get_success_flags(env)
        if success_flags is None:
            success_flags = np.zeros_like(dones_np, dtype=bool)

        for env_i in range(env.unwrapped.num_envs):
            obs_flat = _flatten_obs_to_vector(_tree_index_to_numpy(prev_raw_obs, env_i))
            next_obs_flat = _flatten_obs_to_vector(_tree_index_to_numpy(env.last_raw_obs, env_i))
            transition = {
                "observations": {"state": obs_flat},
                "actions": actions_np[env_i].copy(),
                "next_observations": {"state": next_obs_flat},
                "rewards": float(rewards_np[env_i]),
                "masks": float(1.0 - dones_np[env_i]),
                "dones": bool(dones_np[env_i]),
                "terminated": bool(terminated_np[env_i]),
                "truncated": bool(truncated_np[env_i]),
                "success": bool(success_flags[env_i]),
            }
            transitions.append(transition)
            if dones_np[env_i]:
                episodes += 1

        total_steps += 1
        if isinstance(obs, dict):
            obs = obs.get("obs", obs)

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()

    with open(out_path, "wb") as f:
        pkl.dump(transitions, f)
    print(f"[INFO] Saved {len(transitions)} transitions to {out_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
