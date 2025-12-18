from isaaclab.app.app_launcher import AppLauncher

# ✅ Launch Isaac Sim *first*
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app


import numpy as np
import gymnasium as gym
import torch
from omegaconf import OmegaConf
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
from isaaclab_rl.rl_games import RlGamesVecEnvWrapper, RlGamesGpuEnv
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
import isaaclab_tasks  # registers tasks
from isaaclab_tasks.direct.factory.factory_env_cfg import FactoryEnvCfg, FactoryTaskPegInsertCfg, OBS_DIM_CFG, STATE_DIM_CFG


agent_cfg = OmegaConf.to_container(
    OmegaConf.load("IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/factory/agents/rl_games_ppo_cfg.yaml"),
    resolve=True,
)
agent_cfg["params"]["load_checkpoint"] = True
agent_cfg["params"]["load_path"] = "logs/rl_games/Factory/test/nn/last_Factory_ep_200_rew_355.1727.pth"  # <-- set this
agent_cfg["params"]["config"]["num_actors"] = 1  # temp until real env exists

# dummy spaces from FactoryEnv cfg
cfg_dummy = FactoryEnvCfg()
obs_dim = sum(OBS_DIM_CFG[o] for o in cfg_dummy.obs_order) + cfg_dummy.action_space
state_dim = sum(STATE_DIM_CFG[s] for s in cfg_dummy.state_order) + cfg_dummy.action_space
act_dim = cfg_dummy.action_space
obs_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)
state_space = gym.spaces.Box(-np.inf, np.inf, (state_dim,), dtype=np.float32)
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
agent_cfg["params"]["config"]["num_actors"] = 2
runner = Runner()
runner.load(agent_cfg)
player = runner.create_player()
player.restore(agent_cfg["params"]["load_path"])
player.reset()


records = torch.load("script_peg_insert/state_records.pt")  
reference = []
traj1 = records[0]
reference.append(traj1)  






cfg = FactoryTaskPegInsertCfg()
cfg.scene.num_envs = 1  # set desired num envs
env = gym.make("Isaac-Factory-PegInsert-Direct-v0", cfg=cfg, reference=reference)
if isinstance(env.unwrapped, DirectMARLEnv):
    env = multi_agent_to_single_agent(env)
env = env.unwrapped



obs,info= env.reset()
obs_t = obs['policy']
obs_t = player.obs_to_torch(obs_t)


bs = obs_t.shape[0]
print("batch size is :", bs)
player.get_batch_size(obs_t, bs)
player.init_rnn()
for step in range(1000):
    obs_t = obs['policy']
    obs_t = player.obs_to_torch(obs_t)
    print("fed Observation:", obs_t)    
    action =  player.get_action(obs_t)

    obs, rew, term, trunc, info = env.step(action)
    if term.any().item():
        env.reset()
    
        

        
env.close()

