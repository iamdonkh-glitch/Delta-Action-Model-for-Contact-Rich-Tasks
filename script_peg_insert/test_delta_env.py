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

from isaaclab_tasks.direct.factory.factory_env_cfg import FactoryTaskPegInsertCfg_NoRand
from isaaclab_tasks.direct.factory.factory_env_cfg import FactoryTaskPegInsertCfg

# load recorded reference trajectories
# ref_path = "script_peg_insert/state_records.pt"
# reference_trajs = torch.load(ref_path)
# print(f"Loaded {len(reference_trajs)} reference trajectories from {ref_path}")

records = torch.load("script_peg_insert/state_records.pt")  # list of {"step": ..., "state": {...}}
reference = []
traj1 = records[0]
reference.append(traj1)  # first traj only

# create env
cfg = FactoryTaskPegInsertCfg_NoRand()
cfg.scene.num_envs = 2
env = gym.make("Isaac-Factory-PegInsert-Delta-Direct-v0", cfg=cfg, reference=reference)
env = env.unwrapped

obs, info = env.reset()
for step in range(1000):
    # sample random action
    action = env.action_space.sample()
    #print ("Action:", action)
    # action_space is already batched (num_envs, action_dim); keep shape (1, 6)
    action = torch.tensor(action, device=env.device)
    # step the sim
    obs, rew, term, trunc, info = env.step(action)
    print("Action:", action)
    if term.any().item() or trunc.any().item() or step%50 == 0:

        # joint_pos = torch.tensor([[-0.2432,  0.4370,  0.2486, -2.0322, -0.1641,  2.4508,  0.8951,  0.0040,0.0040]], device=env.device)
        # joint_vel = 0* torch.tensor([[-1.2188e+00,  1.1636e-01,  1.2498e+00,  6.0766e-03, -8.4078e-01,
        #  -7.3975e-02,  5.6004e-01,  5.0424e-06,  7.1064e-06]], device=env.device)
        # env.set_reset_robot_pose(joint_pos, joint_vel)
        # held_pos = torch.tensor([[5.9958e-01, 4.0278e-05, 8.9988e-02]], device=env.device)
        # held_quat = torch.tensor([[-3.2815e-05, -9.7977e-05, -3.3006e-05,  1.0000e+00]], device=env.device)
        # fixed_pos = torch.tensor([[0.6000, 0.0000, 0.0500]], device=env.device)
        # fixed_quat = torch.tensor([[1., 0., 0., 0.]], device=env.device)
        # fixed_pose = {"pos": fixed_pos, "quat": fixed_quat}
        # held_pose = {"pos": held_pos, "quat": held_quat}
        # env.set_reset_asset_pose(fixed_pose, held_pose)
        obs, info = env.reset()
        rob_state = env.get_robot_state()
        ass_state = env.get_asset_state()

        print("Reset Robot State:", rob_state)
        print("Reset Asset State:", ass_state)

env.close()
simulation_app.close()
