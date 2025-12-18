# ================================================================
# keyboard_peginsert_record_demo.py — Keyboard Teleop + Manual Demo Recording
# ================================================================
# 用法：
# ./isaaclab.sh -p custom/keyboard_peginsert_record_demo.py \
#   --task Isaac-Factory-PegInsert-Direct-v0 \
#   --num_envs 1 --device cuda \
#   --dataset_file ./datasets/keyboard_demo.hdf5 \
#   --max_episodes 5
# ================================================================

import argparse
import sys
import time
import os
import torch
import numpy as np
import h5py
import copy
from isaaclab.app import AppLauncher
import datetime
from tqdm import tqdm
#########################################################
import gymnasium as gym
from gymnasium.spaces import flatten_space, flatten



#########################################################
STATE_POLICY_DIM = 19
STATE_CRITIC_DIM = 43
IMAGE_SHAPE = (128, 128, 3)
ACTION_DIM = 6



class FactoryObsToSERLFormatWrapper(gym.ObservationWrapper):
    """
    Converts Factory environment observations to SERL-compatible format.
    Transforms: {"policy", "critic", "camera"} -> {"image": image, "state": flattened}
    Final format: {"image": (128, 128, 3), "state": (62,)}
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # Define observation space
        # ✅ image BEFORE state (matching SERL format)
        self.observation_space = gym.spaces.Dict({
            # "image": gym.spaces.Box(
            #     low=0, high=255,
            #     shape=IMAGE_SHAPE,  # (128, 128, 3)
            #     dtype=np.uint8
            # ),
            "state": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(STATE_POLICY_DIM + STATE_CRITIC_DIM,),  # (62,)
                dtype=np.float32
            ),
        })
        
    def observation(self, obs):
        """Transform Factory obs format to flattened SERL format."""
        if not isinstance(obs, dict) or "policy" not in obs:
            print(f"[WARN] Unexpected obs type: {type(obs)}, keys: {obs.keys() if isinstance(obs, dict) else 'N/A'}")
            return obs
        
        # Extract and convert policy
        if isinstance(obs["policy"], torch.Tensor):
            policy = np.asarray(obs["policy"].detach().cpu().numpy(), dtype=np.float32)
        else:
            policy = np.asarray(obs["policy"], dtype=np.float32)
        
        # Extract and convert critic
        if isinstance(obs["critic"], torch.Tensor):
            critic = np.asarray(obs["critic"].detach().cpu().numpy(), dtype=np.float32)
        else:
            critic = np.asarray(obs["critic"], dtype=np.float32)
        
        # Extract and convert image
        # if isinstance(obs["camera"], torch.Tensor):
        #     image = np.asarray(obs["camera"].detach().cpu().numpy(), dtype=np.uint8)
        # else:
        #     image = np.asarray(obs["camera"], dtype=np.uint8)
        
        # Remove batch dimension if present (convert (1, D) -> (D,))
        if policy.ndim > 1:
            policy = policy[0]
        if critic.ndim > 1:
            critic = critic[0]
        # if image.ndim > 3:
        #     image = image[0]
        
        # Flatten state: concatenate policy + critic
        state = np.concatenate([policy, critic], axis=-1).astype(np.float32)
        
        # ✅ Return with image BEFORE state
        return {
            # "image": image,      # shape: (128, 128, 3)
            "state": state,      # shape: (62,) = (19,) + (43,)
        }

######################################################################################
# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------
parser = argparse.ArgumentParser(description="Keyboard teleop + manual demo recording for Factory PegInsert task.")
parser.add_argument("--task", type=str, default="Isaac-Factory-PegInsert-Direct-v0")
parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=2000)
parser.add_argument("--sleep", type=float, default=0.02)
parser.add_argument("--dataset_file", type=str, default="./datasets/keyboard_demo.hdf5")
parser.add_argument("--max_episodes", type=int, default=5)

AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
setattr(args_cli, "enable_cameras", True)
sys.argv = [sys.argv[0]] + hydra_args

# -------------------------------------------------------------
# Step 1️⃣ 启动 Omniverse App BEFORE importing omni modules
# -------------------------------------------------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -------------------------------------------------------------
# Step 2️⃣ import 一切依赖 Omni 的模块
# -------------------------------------------------------------
import omni
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg
from isaaclab_rl.rl_games import RlGamesVecEnvWrapper
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
import pickle as pkl 

# -------------------------------------------------------------
# 工具类：RateLimiter
# -------------------------------------------------------------
class RateLimiter:
    def __init__(self, hz: int):
        self.hz = max(int(hz), 1)
        self.last_time = time.time()
        self.sleep_duration = 1.0 / self.hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env):
        next_wakeup = self.last_time + self.sleep_duration
        while time.time() < next_wakeup:
            time.sleep(self.render_period)
            try:
                env.sim.render()
            except Exception:
                pass
        self.last_time += self.sleep_duration
        now = time.time()
        if self.last_time < now:
            self.last_time = now


# -------------------------------------------------------------
# 工具函数
# -------------------------------------------------------------
# def step_compat(env, action):
#     """兼容 Gymnasium / IsaacLab step 返回值(4/5元组)"""
#     try:
#         return env.step(action)
#     except ValueError:
#         obs, rew, done, info= env.step(action)
#         return obs, rew, done, truncated, info

def step_compat(env, action):
    """Handle both 4-tuple and 5-tuple step returns."""
    result = env.step(action)
    if len(result) == 5:
        # Gymnasium format: obs, reward, terminated, truncated, info
        return result
    else:
        # Old gym format: obs, reward, done, info
        obs, reward, done, info = result
        return obs, reward, done, False, info  # Add truncated=False


def stabilize_zero_action(env, device):
    """让当前机械臂姿态成为 zero-action 的稳态位姿"""
    try:
        robot = env.unwrapped.scene["robot"]
        for _ in range(5):
            env.unwrapped.sim.step(render=False)
        qpos = robot.data.joint_pos.clone()
        try:
            robot.write_joint_state(qpos, torch.zeros_like(robot.data.joint_vel))
        except Exception:
            robot.set_joint_positions(qpos)
        env.unwrapped.sim.step(render=False)
        print("[INIT] Zero-action aligned to current pose.")
    except Exception as e:
        print("[WARN] Failed to align zero-action:", e)

def flatten_obs(obs):
    """Flatten dict/nested-dict observation into 1D numpy array"""
    if isinstance(obs, dict):
        parts = []
        for v in obs.values():
            parts.append(flatten_obs(v))
        return np.concatenate(parts, axis=-1)
    elif isinstance(obs, torch.Tensor):
        return obs.detach().cpu().numpy().reshape(-1)
    elif isinstance(obs, np.ndarray):
        return obs.reshape(-1)
    else:
        # scalar or list
        return np.array(obs, dtype=np.float32).reshape(-1)

# -------------------------------------------------------------
# Step 3️⃣ 主函数
# -------------------------------------------------------------
#@hydra_task_config(args_cli.task, args_cli.agent)
def main(): #env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: dict)
    import sys
    from pathlib import Path
    isaaclab_root = Path(__file__).parent.parent
    if str(isaaclab_root) not in sys.path:
        sys.path.insert(0, str(isaaclab_root))
    
    from apps.serl_bridge.peg_insert_env import SerlFactoryPegInsertEnv, SerlFactoryTaskPegInsertCfg
    print(f"[INFO] Launching {args_cli.task} with Hydra.")

    #env_cfg.scene.num_envs = args_cli.num_envs
    #env_cfg.sim.device = args_cli.device
 
    # 环境创建
    #env = gym.make(args_cli.task, cfg=env_cfg)
    # try:
    #      env = RlGamesVecEnvWrapper(env, env_cfg.sim.device, clip_obs=float("inf"), clip_actions=float("inf"))
    # except TypeError:
    #      env = RlGamesVecEnvWrapper(env, env_cfg.sim.device, clip_obs=float("inf"), clip_actions=float("inf"), obs_groups=None)
       # ✅ Use the Hydra-provided env_cfg instead of creating a new one
    # This ensures all configurations are properly loaded


    env_cfg = SerlFactoryTaskPegInsertCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    #print("env_cfg is ", env_cfg)
 
    # Now create the environment with Hydra-loaded config
    env = SerlFactoryPegInsertEnv(cfg=env_cfg)
    env = FactoryObsToSERLFormatWrapper(env)
    
    
    obs, _ = env.reset()
    #stabilize_zero_action(env, args_cli.device)

    rate = RateLimiter(int(1 / args_cli.sleep))
    teleop = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.2, rot_sensitivity=0.5))
    print("[INFO] Keyboard teleop ready: WASD/EQ (move) | IJKL/UO (rotate)")

    os.makedirs(os.path.dirname(args_cli.dataset_file), exist_ok=True)
    h5_path = args_cli.dataset_file
    print(f"[REC] Saving demos to {h5_path}")


    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_dir = os.path.dirname(os.path.abspath(args_cli.dataset_file))
    file_name = f"peg_insert_{args_cli.max_episodes}_demos_{uuid}.pkl"
    file_path = os.path.join(file_dir, file_name)

    # ---------------------------------------------------------
    # 主循环
    # ---------------------------------------------------------
    transitions = []
    
    success_count = 0
    episode_i = 1
    while simulation_app.is_running() :
        teleop = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.2, rot_sensitivity=0.5))
        curr_transitions = []
        step_i = 0
        print(f"[INFO] Recording episode #{episode_i} ...")

        while simulation_app.is_running():
            delta_pose = teleop.advance()
            action = torch.clamp(delta_pose[0:6], -1.0, 1.0)
            if isinstance(action, torch.Tensor):
                actions_np = action.cpu().numpy().astype(np.float32)
            else:
                actions_np = np.array(action, dtype=np.float32)

            next_obs, rew, done,truncated, info = step_compat(env, action)
            #print("content of obs:", obs)
            #print("content of done:", done)

            # Convert done to proper scalar/float for masks calculation
            if isinstance(done, torch.Tensor):
                done_val = float(done.detach().cpu().item())
                #print("done is tensor")
            else:
                done_val = float(done)
            #print("obs ", obs)
            #print("obs_image shape:", next_obs['camera'].shape)
            #print("obs_state shape:", next_obs['state'].shape)
            rew = float(rew.detach().cpu().item()) if isinstance(rew, torch.Tensor) else float(rew)
            done = bool(done) if isinstance(done, (bool, torch.Tensor)) else bool(done)
            done_val = float(done_val.detach().cpu().item()) if isinstance(done_val, torch.Tensor) else float(done_val)
            #print("obs type:", type(obs))
            #print("next_obs type:", type(next_obs))
            assert isinstance(rew, float), "Reward must be a float"
            assert isinstance(done, bool), "Done must be a bool"
            assert isinstance(done_val, float), "done_val must be a float"
            transition = copy.deepcopy(
                    dict(
                    observations=obs,
                    actions=actions_np,
                    next_observations=next_obs,
                    rewards=rew,
                    masks=1.0 - done,
                    dones=done,
                )
            )
            curr_transitions.append(transition)


            # if step_i % 10 == 0:
            #     print(f"[Ep {episode_i} | {step_i:04d}] reward={float(np.mean(rew)):.4f}")
            step_i += 1
            obs = next_obs
            rate.sleep(env)
            
            if done:
                # ask whether save curr_transitions
                save_input = input("Save this episode? (y/n): ").strip().lower()
                if save_input == 'y':
                    transitions.extend(curr_transitions)
                    success_count += 1
                    print(f"[SAVE] Episode #{episode_i} saved. Total demos: {len(transitions)} transitions. Success count: {success_count}.")
                break

       

        episode_i += 1
        if success_count > args_cli.max_episodes:
            break
        obs, _ = env.reset()
        #stabilize_zero_action(env, args_cli.device)
        
    env.close()


    with open(file_path, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {len(transitions)} demos to {file_path}")


    print(f"[✅] Teleop + Manual Recording finished. Demos saved to {h5_path}")


# -------------------------------------------------------------
# Step 4️⃣ 运行
# -------------------------------------------------------------
if __name__ == "__main__":
    main()
    simulation_app.close()
