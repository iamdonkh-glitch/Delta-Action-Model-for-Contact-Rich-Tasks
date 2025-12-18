from isaaclab.app.app_launcher import AppLauncher

# ✅ Launch Isaac Sim *first*
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import numpy as np
from omegaconf import OmegaConf
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
from isaaclab_rl.rl_games import RlGamesVecEnvWrapper, RlGamesGpuEnv
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
import isaaclab_tasks  # registers tasks
from isaaclab_tasks.direct.factory.factory_env_cfg import FactoryTaskPegInsertCfg_NoRand

# load agent cfg and checkpoint
agent_cfg = OmegaConf.to_container(
    OmegaConf.load("IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/factory/agents/rl_games_ppo_cfg.yaml"),
    resolve=True,
)
agent_cfg["params"]["load_checkpoint"] = True
agent_cfg["params"]["load_path"] = "logs/rl_games/Factory/test/nn/last_Factoryep200rew[355.1727].pth"

# build env: 64 parallel peg-insert direct
cfg = FactoryTaskPegInsertCfg_NoRand()
cfg.scene.num_envs = 64
env = gym.make("Isaac-Factory-PegInsert-Direct-v0", cfg=cfg)
if isinstance(env.unwrapped, DirectMARLEnv):
    env = multi_agent_to_single_agent(env)

# wrap for rl-games (like train/play)
rl_device = agent_cfg["params"]["config"]["device"]
clip_obs = agent_cfg["params"]["env"].get("clip_observations", np.inf)
clip_actions = agent_cfg["params"]["env"].get("clip_actions", np.inf)
obs_groups = agent_cfg["params"]["env"].get("obs_groups")
concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)
env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs))
env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
runner = Runner()
runner.load(agent_cfg)
player = runner.create_player()
player.restore(agent_cfg["params"]["load_path"])
player.reset()

# rollout and count successes
num_episodes = 1
successes = []
obs = env.reset()
# required: enable batched obs flag like play.py
_ = player.get_batch_size(obs["obs"] if isinstance(obs, dict) else obs, env.unwrapped.num_envs)
if player.is_rnn:
    player.init_rnn()

for ep in range(num_episodes):
    done = torch.zeros(env.unwrapped.num_envs, dtype=torch.bool, device=rl_device)
    while not done.all():
        obs_t = obs["obs"] if isinstance(obs, dict) else obs
        obs_t = player.obs_to_torch(obs_t)
        actions = player.get_action(obs_t)
        obs, rew, done, extras = env.step(actions)
        if "successes" in extras:
            successes.append(float(extras["successes"]))
    obs = env.reset()

print(f"Success rate per episode ({len(successes)} eps): {successes}")
print(f"Mean success: {np.mean(successes) if successes else 0.0}")
env.close()
simulation_app.close()
