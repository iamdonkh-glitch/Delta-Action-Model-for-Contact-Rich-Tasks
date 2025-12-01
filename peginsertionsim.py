# peginsertionsim.py

from isaaclab.app.app_launcher import AppLauncher

# ✅ Launch Isaac Sim *first*
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

# 🚫 Do NOT import IsaacLab/Omni modules before this point
# -------------------------------------------------------

# Now safe to import the rest
import gymnasium as gym
import isaaclab_tasks  # registers tasks with gym

from isaaclab_tasks.direct.factory.factory_env_cfg import FactoryTaskPegInsertCfg

# create env
cfg = FactoryTaskPegInsertCfg()
cfg.scene.num_envs = 1
env = gym.make("Isaac-Factory-PegInsert-Direct-v0", cfg=cfg)
env = env.unwrapped
import torch

obs, info = env.reset()
for _ in range(1000):
    # sample random action
    action = env.action_space.sample()
    print ("Action:", action)
    # convert numpy → torch tensor (IsaacLab expects tensors)
    action = torch.tensor(action, device=env.device)
    # step the sim
    obs, rew, term, trunc, info = env.step(action)
    if term or trunc:
        obs, info = env.reset()

env.close()
simulation_app.close()
