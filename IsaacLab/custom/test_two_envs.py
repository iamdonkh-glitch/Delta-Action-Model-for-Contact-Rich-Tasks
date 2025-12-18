# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Run a trained RL-Games PPO checkpoint to record demos (no keyboard teleop).

Example:
./isaaclab.sh -p custom/test_two_envs.py \
    --task Isaac-Factory-PegInsert-Direct-v0 \
    --num_envs 2 \
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
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Ensure repo root is on path so we can import apps.*
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from isaaclab.app import AppLauncher

# CLI
parser = argparse.ArgumentParser(description="Collect demos using a trained RL-Games PPO agent (no teleop).")
parser.add_argument("--task", type=str, default="Isaac-Factory-PegInsert-Direct-v0")
parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--num_steps", type=int, default=2000)
parser.add_argument(
    "--trajectories",
    type=int,
    default=None,
    help="Stop after this many trajectories (episodes) across all envs; optional.",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="./logs/rl_games/Factory/test/nn/last_Factory_ep_200_rew_365.6033.pth",
    help="Path to RL-Games checkpoint.",
)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--output", type=str, default="./datasets/test_two_envs_demos.pkl")
parser.add_argument("--real-time", action="store_true", default=False, help="Sleep to match sim dt.")

AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
setattr(args_cli, "enable_cameras", True)
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports that require the app to be running
import gymnasium as gym
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from apps.serl_bridge.peg_insert_env import SerlFactoryPegInsertEnv, SerlFactoryTaskPegInsertCfg


class RecordingRlGamesVecEnvWrapper(RlGamesVecEnvWrapper):
    """RlGames wrapper that keeps raw observations and actions for logging."""

    def reset(self):
        obs_dict, _ = self.env.reset()
        self.last_raw_obs = obs_dict
        self.last_actions = None
        self.last_terminated = None
        self.last_time_outs = None
        return self._process_obs(obs_dict)

    def step(self, actions):
        actions = actions.detach().clone().to(device=self._sim_device)
        actions = torch.clamp(actions, -self._clip_actions, self._clip_actions)
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)

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
    if isinstance(data, dict):
        return {k: _tree_index_to_numpy(v, idx) for k, v in data.items()}
    if isinstance(data, torch.Tensor):
        arr = data.detach().cpu().numpy()
        return arr[idx].copy() if arr.ndim > 0 else arr.copy()
    if isinstance(data, np.ndarray):
        return data[idx].copy() if data.ndim > 0 else data.copy()
    return data


def _flatten_obs_to_vector(obs: Any) -> np.ndarray:
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
    return None


def _resolve_checkpoint(agent_cfg: dict, task_name: str) -> str:
    log_root_path = os.path.abspath(os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"]))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.checkpoint and os.path.exists(args_cli.checkpoint):
        return retrieve_file_path(args_cli.checkpoint)
    if args_cli.checkpoint and not os.path.exists(args_cli.checkpoint):
        print(f"[WARN] Provided checkpoint not found: {args_cli.checkpoint}")
    run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
    resume_path = get_checkpoint_path(log_root_path, run_dir, ".*", other_dirs=["nn"])
    return resume_path


def _timestamped_path(base_path: str) -> str:
    base_path = os.path.abspath(base_path)
    folder, fname = os.path.split(base_path)
    stem, ext = os.path.splitext(fname)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{stem}_{ts}{ext}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.device is not None:
        agent_cfg["params"]["config"]["device"] = args_cli.device
        agent_cfg["params"]["config"]["device_name"] = args_cli.device

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    env_cfg.seed = agent_cfg["params"]["seed"]

    resume_path = _resolve_checkpoint(agent_cfg, train_task_name)
    log_dir = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir

    # Create SERL env
    serl_cfg = SerlFactoryTaskPegInsertCfg()
    serl_cfg.scene.num_envs = args_cli.num_envs
    serl_cfg.sim.device = args_cli.device
    env = SerlFactoryPegInsertEnv(cfg=serl_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

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
    target_episodes = args_cli.trajectories
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
