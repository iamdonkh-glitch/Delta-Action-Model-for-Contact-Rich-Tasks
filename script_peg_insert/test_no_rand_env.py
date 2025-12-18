from isaaclab.app.app_launcher import AppLauncher

# ✅ Launch Isaac Sim *first*
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

# 🚫 Do NOT import IsaacLab/Omni modules before this point
# -------------------------------------------------------
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
from isaaclab_tasks.direct.factory import factory_utils

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


records = torch.load("script_peg_insert/state_record.pt")  
reference = []
traj1 = records[0]
reference.append(traj1)  


# Now safe to import the rest
import gymnasium as gym
import isaaclab_tasks  # registers tasks with gym

from isaaclab_tasks.direct.factory.factory_env_cfg import FactoryTaskPegInsertCfg_NoRand
from isaaclab_tasks.direct.factory.factory_env_cfg import FactoryTaskPegInsertCfg
# create env
cfg = FactoryTaskPegInsertCfg_NoRand()
cfg.scene.num_envs = 1
env = gym.make("Isaac-Factory-PegInsert-NoRand-Direct-v0", cfg=cfg)
env = env.unwrapped
import torch
# step 0 robot state: {'joint_pos': tensor([[-0.0470,  0.5371,  0.0252, -2.0189, -0.0233,  2.5558,  1.5387,  0.0050,
#           0.0050]], device='cuda:0'), 'joint_vel': tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0.]], device='cuda:0'), 'fingertip_pos': tensor([[ 0.6014, -0.0130,  0.0689]], device='cuda:0'), 'fingertip_quat': tensor([[-3.4689e-06,  9.2881e-01, -3.7055e-01,  1.3008e-06]], device='cuda:0')}
#Setting to custom reset asset pose fixed state and held state are: {'pos': tensor([[0.5899, 0.0017, 0.0025]], device='cuda:0'), 'quat': tensor([[0.0971, 0.0000, 0.0000, 0.9953]], device='cuda:0')} {'pos': tensor([[ 0.6016, -0.0132,  0.0390]], device='cuda:0'), 'quat': tensor([[ 3.7055e-01, -1.3113e-06, -3.4571e-06,  9.2881e-01]], device='cuda:0')}
#ref action at step  tensor([0], device='cuda:0')  is  tensor([[-1.0000,  1.0000, -1.0000,  1.0000,  0.6745,  0.4493]],
       #device='cuda:0')

joint_pos = torch.tensor(
    [[-0.0470, 0.5371, 0.0252, -2.0189, -0.0233, 2.5558, 1.5387, 0.0050, 0.0050]],
    device=env.device,
)
joint_vel = torch.zeros_like(joint_pos)

fixed_pose = {
    "pos": torch.tensor([[0.5899, 0.0017, 0.0025]], device=env.device),
    "quat": torch.tensor([[0.0971, 0.0000, 0.0000, 0.9953]], device=env.device),
}
held_pose = {
    "pos": torch.tensor([[0.6016, -0.0132, 0.0390]], device=env.device),
    "quat": torch.tensor([[ 3.7055e-01, -1.3113e-06, -3.4571e-06,  9.2881e-01]], device=env.device),
}

env.set_reset_robot_pose(joint_pos, joint_vel)
env.set_reset_asset_pose(fixed_pose, held_pose)

# env.task_prop_gains = env.default_gains.clone()
# env.task_deriv_gains = factory_utils.get_deriv_gains(env.default_gains)
# env.actions = torch.zeros_like(env.actions)
# env.prev_actions = torch.zeros_like(env.actions)


obs, info = env.reset()
obs_t = obs['policy']
obs_t = player.obs_to_torch(obs_t)

for step in range(5):
    # sample random action
    obs_t = obs['policy']
    obs_t = player.obs_to_torch(obs_t)
    #print("fed Observation:", obs_t)
    #action =  player.get_action(obs_t)
    action = torch.tensor([[-1.0000,  1.0000, -1.0000,  1.0000,  0.6745,  0.4493]],
         device='cuda:0')

    # action = torch.tensor(action, device=env.device)
    # step the sim
    obs, rew, term, trunc, info = env.step(action)
    rob_state = env.get_robot_state()
    ass_state = env.get_asset_state()

    print("Robot State at step", step, ":", rob_state)

    if term or trunc or step%10 == 0:
         ##Reset RobotState: {'joint_pos': tensor([[-0.2432,  0.4370,  0.2486, -2.0322, -0.1641,  2.4508,  0.8951,  0.0040,
          ##0.0040]], device='cuda:0'), 'joint_vel': tensor([[-1.2188e+00,  1.1636e-01,  1.2498e+00,  6.0766e-03, -8.4078e-01,
         ##-7.3975e-02,  5.6004e-01,  5.0424e-06,  7.1064e-06]], device='cuda:0'), 'fingertip_pos': tensor([[5.9957e-01, 3.2086e-05, 1.2237e-01]], device='cuda:0'), 'fingertip_quat': tensor([[-3.5511e-05,  1.0000e+00,  4.3499e-05,  7.8732e-05]], device='cuda:0')}
        #Reset Asset State: ({'fixed_pos': tensor([[0.6000, 0.0000, 0.0500]], device='cuda:0'), 'fixed_quat': tensor([[1., 0., 0., 0.]], device='cuda:0')}, {'held_pos': tensor([[5.9958e-01, 4.0278e-05, 8.9988e-02]], device='cuda:0'), 'held_quat': tensor([[-3.2815e-05, -9.7977e-05, -3.3006e-05,  1.0000e+00]], device='cuda:0')})
        # Setting to custom reset asset pose fixed state and held state are: {'pos': tensor([[0.5899, 0.0017, 0.0025]], device='cuda:0'), 'quat': tensor([[0.0971, 0.0000, 0.0000, 0.9953]], device='cuda:0')} {'pos': tensor([[ 0.6005, -0.0128,  0.0394]], device='cuda:0'), 'quat': tensor([[ 3.6968e-01, -5.9854e-06, -5.0726e-04,  9.2916e-01]], device='cuda:0')}
        # Setting to custom reset pose joint pos and joint vel is: tensor([[-0.3272,  0.5596,  0.3042, -2.0160, -0.2828,  2.5355,  1.7103,  0.0000,
        #   0.0000]], device='cuda:0') tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0.]], device='cuda:0')



        obs, info = env.reset()
        rob_state = env.get_robot_state()
        ass_state = env.get_asset_state()

        print("Reset Robot State:", rob_state)
        print("Reset Asset State:", ass_state)

env.close()
simulation_app.close()
