"""Run a trained agent for one episode and record Factory state dicts each step."""

import argparse
import math
import os
import random
import sys
import time

import gymnasium as gym
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.app import AppLauncher



# -----------------------------------------------------------------------------#
# CLI
# -----------------------------------------------------------------------------#
parser = argparse.ArgumentParser(description="Record per-step state dicts for one episode.")
parser.add_argument("--task", type=str, default="Isaac-Factory-PegInsert-Direct-v0", help="Name of the task to run.")
parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point", help="Hydra entry point for the agent.")
parser.add_argument("--checkpoint", type=str, default=None, help="Explicit path to a model checkpoint.")
parser.add_argument(
    "--use_pretrained_checkpoint", action="store_true", help="Use the published pretrained checkpoint if available."
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model; otherwise use the best model.",
)
parser.add_argument("--seed", type=int, default=None, help="Seed for the rollout; -1 samples a random seed.")
parser.add_argument("--real-time", dest="real_time", action="store_true", help="Run in real-time if possible.")
parser.add_argument(
    "--max_steps", type=int, default=None, help="Optional hard cap on rollout steps (otherwise runs to episode end)."
)
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Path to save recorded trajectories (.pt). Defaults to the checkpoint log directory.",
)
parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to record.")

# add AppLauncher CLI args and launch sim
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

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
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401


# -----------------------------------------------------------------------------#
# Main
# -----------------------------------------------------------------------------#
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Load the agent (mirrors eval_success) and record state dicts for one episode."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # force a single environment for this recording script
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    env_cfg.seed = agent_cfg["params"]["seed"]

    # locate checkpoint (same logic as eval_success)
    log_root_path = os.path.abspath(os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"]))
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rl_games", train_task_name)
        if not resume_path:
            print("[INFO] Pre-trained checkpoint unavailable for this task.")
            return
    elif args_cli.checkpoint is None:
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        checkpoint_file = ".*" if args_cli.use_last_checkpoint else f"{agent_cfg['params']['config']['name']}.pth"
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    if resume_path is None:
        raise FileNotFoundError("Unable to locate a checkpoint to load.")
    log_dir = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir
    print(f"[INFO] Loading model checkpoint from: {resume_path}")

    # create environment and rl-games wrapper
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RlGamesVecEnvWrapper(
        env, agent_cfg["params"]["config"]["device"], clip_obs, clip_actions, obs_groups, concate_obs_groups
    )

    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    dt = env.unwrapped.step_dt
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs.get("obs", obs)

    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    trajectories = []
    episodes_run = 0
    max_steps = args_cli.max_steps

    while simulation_app.is_running() and episodes_run < args_cli.num_episodes:
        traj = []  # one trajectory = list of per-step state dicts
        step_idx = 0
        while simulation_app.is_running():
            with torch.no_grad():
                obs_torch = agent.obs_to_torch(obs)
                actions = agent.get_action(obs_torch)
                obs, _, dones, _ = env.step(actions)

                # grab current state dict and append
                _, state_dict, _ = env.unwrapped._get_factory_obs_state_dict()
                traj.append({k: v.clone().cpu() for k, v in state_dict.items()})

                step_idx += 1
                if len(dones) > 0 and torch.any(dones):
                    if agent.is_rnn and agent.states is not None:
                        for s in agent.states:
                            s[:, dones, :] = 0.0
                    episodes_run += 1
                    trajectories.append(traj)
                    break

                if max_steps is not None and step_idx >= max_steps:
                    print(f"[WARN] Episode {episodes_run + 1}: reached max_steps={max_steps} before episode end.")
                    episodes_run += 1
                    trajectories.append(traj)
                    break

            # real-time sleep logic stays here if needed

        if episodes_run < args_cli.num_episodes:
            obs = env.reset()
            if isinstance(obs, dict):
                obs = obs.get("obs", obs)

    output_path = args_cli.output or os.path.join(log_dir, "state_trajectories.pt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(trajectories, output_path)
    print(f"[INFO] Saved {len(trajectories)} trajectories to {output_path}")



if __name__ == "__main__":
    main()
    simulation_app.close()
