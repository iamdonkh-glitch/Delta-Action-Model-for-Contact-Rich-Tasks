from isaaclab.app.app_launcher import AppLauncher

# ✅ Launch Isaac Sim *first*
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

# 🚫 Do NOT import IsaacLab/Omni modules before this point
# -------------------------------------------------------

# Now safe to import the rest
import gymnasium as gym
import isaaclab_tasks  # registers tasks with gym
import torch
import numpy as np
from rl_games.common.player import BasePlayer
from isaaclab_tasks.direct.factory.factory_env_cfg import FactoryTaskPegInsertCfg_NoRand
from isaaclab_tasks.direct.factory.factory_env_cfg import FactoryTaskPegInsertCfg
from omegaconf import OmegaConf
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
from isaaclab_rl.rl_games import RlGamesVecEnvWrapper, RlGamesGpuEnv
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
import isaaclab_tasks  # registers tasks
from isaaclab_tasks.direct.factory.factory_env_cfg import FactoryEnvCfg, FactoryTaskPegInsertCfg, OBS_DIM_CFG, STATE_DIM_CFG
import math
# --- 1) Load agent cfg and create player before real env ---
agent_cfg = OmegaConf.to_container(
    OmegaConf.load("IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/factory/agents/rl_games_ppo_cfg.yaml"),
    resolve=True,
)
agent_cfg["params"]["load_checkpoint"] = True
agent_cfg["params"]["load_path"] = "logs/rl_games/Factory/delta_weighted_xyz/nn/last_Factoryep400rew[445.1203].pth"  # <-- set this
agent_cfg["params"]["config"]["num_actors"] = 1  # temp until real env exists

# dummy spaces from FactoryEnv cfg
cfg_dummy = FactoryEnvCfg()
obs_dim = sum(OBS_DIM_CFG[o] for o in cfg_dummy.obs_order) + cfg_dummy.action_space
state_dim = sum(STATE_DIM_CFG[s] for s in cfg_dummy.state_order) + cfg_dummy.action_space
act_dim = cfg_dummy.action_space
obs_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim+6,), dtype=np.float32)
state_space = gym.spaces.Box(-np.inf, np.inf, (state_dim+6,), dtype=np.float32)
act_space = gym.spaces.Box(-1.0, 1.0, (act_dim,), dtype=np.float32)

class DummyEnv:
    def __init__(self):
        self.observation_space = obs_space
        self.state_space = state_space
        self.action_space = act_space
    def get_env_info(self):
        return {"observation_space": obs_space, "state_space": state_space, "action_space": act_space}

env_configurations.register("dummy", {"env_creator": lambda **kwargs: DummyEnv()})
agent_cfg["params"]["config"]["env_name"] = "dummy"
agent_cfg["params"]["config"]["num_actors"] = 1
runner = Runner()
runner.load(agent_cfg)
player = runner.create_player()
player.restore(agent_cfg["params"]["load_path"])
player.reset()

print("restored!!!")

reference = torch.load("script_peg_insert/state_record_50.pt")
slot = []
slot.append(reference[0])
reference = slot  # use only one traj for this test

# create env
cfg = FactoryTaskPegInsertCfg_NoRand()
cfg.scene.num_envs = 1
env = gym.make("Isaac-Factory-PegInsert-Delta-CloseLoop-v0", cfg=cfg, BaseAgent=player)
env = env.unwrapped
obs, info = env.reset()
print(f"env reset obs: {obs}")
#current_robot_state = env.get_robot_state()
#print(f"step 0 robot state: {current_robot_state}")
episode_returns = torch.zeros(env.num_envs, device=env.device)
completed_returns = []


# guard against shorter refs
# ref_entry = reference[0][0]
# ref_robot_state = {
#                 "joint_pos": ref_entry["joint_pos"],
#                 #"joint_vel": ref_entry["joint_vel"],
#                 "fingertip_pos": ref_entry["fingertip_pos"],
#                 "fingertip_quat": ref_entry["fingertip_quat"],
#             }
# print(f"ref robot state (env {0}, step {0}): {ref_robot_state}")
reward = 0
for step in range(1000):
    # sample random action
    action = 0* env.action_space.sample()
    #action = player.get_action(obs['policy'])
    action = torch.tensor(action, device=env.device)

    obs, rew, term, trunc, info = env.step(action)
    #env.step(action)
    # env.ref_sim(step)
    # rew = env._get_rewards()
    # rew is per-env tensor; accumulate before any reductions
    episode_returns += rew
    print("reward at step ", step+1, " is ", rew.cpu().numpy())
    current_robot_state = env.get_robot_state()
    print(f"step {step+1} robot state: {current_robot_state}")
    

        # Reference robot state for this env and step
    # env_id = 0  # only one env in this script
    # traj_idx = env.unwrapped._ref_traj_ids[env_id].item()
    # if step+1 < len(reference[traj_idx]):  # guard against shorter refs
    #         ref_entry = reference[traj_idx][step+1]
    #         ref_robot_state = {
    #             "joint_pos": ref_entry["joint_pos"],
    #             #"joint_vel": ref_entry["joint_vel"],
    #             "fingertip_pos": ref_entry["fingertip_pos"],
    #             "fingertip_quat": ref_entry["fingertip_quat"],
    #             "fixed_pos": ref_entry["fixed_pos"],
    #         }
    #         print(f"ref robot state (env {env_id}, step {step+1}): {ref_robot_state}")
    # else:
    #         print(f"ref robot state unavailable: step {step} >= len(reference[{traj_idx}])")
   
    #if term.any().item() or trunc.any().item() :
    if step >=149:
        #done = torch.logical_or(term, trunc)
        done = True
        done_ids = torch.nonzero(done, as_tuple=False).flatten().tolist()
        for idx in done_ids:
            completed_returns.append(episode_returns[idx].item())
            episode_returns[idx] = 0.0
        obs, info = env.reset()
        #reset_robot_state = env.get_robot_state()
        #print(f"reset robot state: {reset_robot_state}")
        break

if completed_returns:
    mean_rew = float(np.mean(completed_returns))
    print("Completed episode returns:", completed_returns)
    print("Mean reward per episode:", mean_rew)
else:
    print("No completed episodes; partial returns:", episode_returns.cpu().numpy())

env.close()
simulation_app.close()