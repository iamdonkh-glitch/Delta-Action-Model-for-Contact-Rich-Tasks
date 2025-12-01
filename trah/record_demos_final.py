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

from isaaclab.app import AppLauncher

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
def step_compat(env, action):
    """兼容 Gymnasium / IsaacLab step 返回值(4/5元组)"""
    try:
        return env.step(action)
    except ValueError:
        obs, rew, done, info = env.step(action)
        return obs, rew, done, done, info


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
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: dict):
    print(f"[INFO] Launching {args_cli.task} with Hydra.")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    # 环境创建
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        env = RlGamesVecEnvWrapper(env, env_cfg.sim.device, clip_obs=float("inf"), clip_actions=float("inf"))
    except TypeError:
        env = RlGamesVecEnvWrapper(env, env_cfg.sim.device, clip_obs=float("inf"), clip_actions=float("inf"), obs_groups=None)

    obs, _ = env.reset()
    stabilize_zero_action(env, args_cli.device)

    rate = RateLimiter(int(1 / args_cli.sleep))
    teleop = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.2, rot_sensitivity=0.5))
    print("[INFO] Keyboard teleop ready: WASD/EQ (move) | IJKL/UO (rotate)")

    os.makedirs(os.path.dirname(args_cli.dataset_file), exist_ok=True)
    h5_path = args_cli.dataset_file
    print(f"[REC] Saving demos to {h5_path}")

    # ---------------------------------------------------------
    # 主循环
    # ---------------------------------------------------------
    episode_i = 1
    while simulation_app.is_running() and episode_i <= args_cli.max_episodes:
        ep_obs, ep_actions, ep_rewards, ep_dones = [], [], [], []
        step_i = 0
        print(f"[INFO] Recording episode #{episode_i} ...")

        while simulation_app.is_running():
            delta_pose = teleop.advance()
            action = torch.clamp(delta_pose[0:6], -1.0, 1.0)
            obs, rew, term, info = step_compat(env, action)

            obs = flatten_obs(obs)

            ep_obs.append(torch.tensor(obs).cpu().numpy())
            ep_actions.append(action.cpu().numpy())
            ep_rewards.append(float(rew))
            ep_dones.append(bool(term))

            # if step_i % 10 == 0:
            #     print(f"[Ep {episode_i} | {step_i:04d}] reward={float(np.mean(rew)):.4f}")
            step_i += 1
            rate.sleep(env)

            done = (term[0] if isinstance(term, (list, torch.Tensor)) else term) or step_i >= args_cli.max_steps
            if done:
                break

        # 保存一集
        with h5py.File(h5_path, "a") as f:
            grp = f.create_group(f"episode_{episode_i:03d}")
            grp.create_dataset("obs", data=np.array(ep_obs, dtype=np.float32))
            grp.create_dataset("action", data=np.array(ep_actions, dtype=np.float32))
            grp.create_dataset("reward", data=np.array(ep_rewards, dtype=np.float32))
            grp.create_dataset("done", data=np.array(ep_dones, dtype=np.bool_))
        print(f"[INFO] Episode #{episode_i} saved ({len(ep_obs)} steps).")

        episode_i += 1
        if episode_i > args_cli.max_episodes:
            break
        obs, _ = env.reset()
        stabilize_zero_action(env, args_cli.device)

    env.close()
    print(f"[✅] Teleop + Manual Recording finished. Demos saved to {h5_path}")


# -------------------------------------------------------------
# Step 4️⃣ 运行
# -------------------------------------------------------------
if __name__ == "__main__":
    main()
    simulation_app.close()
