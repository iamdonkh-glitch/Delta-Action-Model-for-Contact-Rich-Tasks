# ================================================================
# keyboard_peginsert_control.py — Hydra + Keyboard Teleop (SE3)
# ================================================================
# 用法：
# ./isaaclab.sh -p custom/keyboard_peginsert_control.py \
#   --task Isaac-Factory-PegInsert-Direct-v0 \
#   --num_envs 1 --device cuda
# ================================================================

import argparse
import sys
import time
import torch
from distutils.util import strtobool

from isaaclab.app import AppLauncher

# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------
parser = argparse.ArgumentParser(description="Keyboard teleop for Factory PegInsert task.")
parser.add_argument("--task", type=str, default="Isaac-Factory-PegInsert-Direct-v0")
parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=2000)
parser.add_argument("--sleep", type=float, default=0.02)
AppLauncher.add_app_launcher_args(parser)

# 解析 CLI
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
# 工具函数
# -------------------------------------------------------------
def step_compat(env, action):
    """兼容 Gymnasium / IsaacLab step 返回值"""
    try:
        return env.step(action)
    except ValueError:
        obs, rew, done, info = env.step(action)
        term, trunc = done, done
        return obs, rew, term, trunc, info


def stabilize_zero_action(env, device):
    """让当前机械臂姿态成为 zero-action 的稳态位姿"""
    try:
        robot = env.unwrapped.scene["robot"]

        for _ in range(5):
            env.unwrapped.sim.step(render=False)

        qpos = robot.data.joint_pos.clone()
        qvel = torch.zeros_like(robot.data.joint_vel)

        robot.write_joint_state(qpos, qvel)
        env.unwrapped.sim.step(render=False)

        # 同步控制器目标
        if hasattr(env.unwrapped, "controllers") and "arm_action" in env.unwrapped.controllers:
            ctrl = env.unwrapped.controllers["arm_action"]
            if hasattr(ctrl, "set_default_target"):
                ctrl.set_default_target(qpos)
                print("[INIT] Controller default target set to current joint pose.")

        # 重置动作缓存
        if hasattr(env.unwrapped, "managers") and "action_manager" in env.unwrapped.managers:
            amgr = env.unwrapped.managers["action_manager"]
            if hasattr(amgr, "reset"):
                amgr.reset()
                print("[INIT] Action manager reset for clean start.")

        print("[INIT] Robot stabilized; zero-action now matches current pose.")
    except Exception as e:
        print("[WARN] Failed to align zero-action with current pose:", e)


# -------------------------------------------------------------
# Step 3️⃣ 主函数（Hydra 装饰器）
# -------------------------------------------------------------
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: dict):
    print(f"[INFO] Launching {args_cli.task} with Hydra.")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    # =========================================================
    # 防抽搐补丁
    # =========================================================
    try:
        env_cfg.actions.arm_action.use_default_offset = False
        print("[PATCH] use_default_offset=False")
    except Exception:
        pass
    try:
        env_cfg.actions.arm_action.scale = 0.8
        print("[PATCH] arm_action.scale=0.2")
    except Exception:
        pass
    try:
        env_cfg.randomization.reset_on_init = False
        print("[PATCH] randomization.reset_on_init=False")
    except Exception:
        pass
    try:
        physx = env_cfg.sim.physx
        physx.solver_position_iteration_count = max(physx.solver_position_iteration_count, 16)
        physx.solver_velocity_iteration_count = max(physx.solver_velocity_iteration_count, 1)
        physx.contact_offset = 0.005
        physx.rest_offset = 0.0
        if hasattr(physx, "enable_stabilization"):
            physx.enable_stabilization = True
        print("[PATCH] PhysX stability increased.")
    except Exception:
        pass

    # =========================================================
    # 创建环境（保持与 train 一致）
    # =========================================================
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        env = RlGamesVecEnvWrapper(env, env_cfg.sim.device,
                                   clip_obs=float("inf"), clip_actions=float("inf"))
    except TypeError:
        env = RlGamesVecEnvWrapper(env, env_cfg.sim.device,
                                   clip_obs=float("inf"), clip_actions=float("inf"),
                                   obs_groups=None)

    obs, _ = env.reset()
    stabilize_zero_action(env, args_cli.device)

    # =========================================================
    # 键盘控制初始化
    # =========================================================
    teleop = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.2, rot_sensitivity=0.5))
    print("[INFO] Keyboard teleop ready: WASD/EQ (move) | IJKL/UO (rotate) | Space (grip toggle)")

    grip_state = 0.0
    step_i = 0

    # =========================================================
    # 主循环
    # =========================================================
    while simulation_app.is_running() and step_i < args_cli.max_steps:
        delta_pose = teleop.advance()  # CPU tensor, 6维 (dx, dy, dz, rx, ry, rz)
        delta_device = delta_pose.device if hasattr(delta_pose, "device") else torch.device("cpu")

  
        # --- 对齐动作空间 ---
        # 拼接 gripper 通道形成 7D
        action = delta_pose[0:6]

       

        # clip 并 step
        action = torch.clamp(action, -1.0, 1.0)
        obs, rew, term, info = step_compat(env, action)

        if step_i % 10 == 0:
            mean_r = torch.mean(torch.tensor(rew)).item()
            print(f"[{step_i:04d}] action={action.cpu().numpy().round(3)} reward={mean_r:.4f}")

        step_i += 1
        time.sleep(args_cli.sleep)

    env.close()
    print("[INFO] Teleop finished.")


# -------------------------------------------------------------
# Step 4️⃣ 运行
# -------------------------------------------------------------
if __name__ == "__main__":
    main()
    simulation_app.close()
