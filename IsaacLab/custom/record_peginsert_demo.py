# ================================================================
# record_peginsert_demo.py — Zero-action rollout (Hydra-based)
# ================================================================
# 目的：加载 Factory PegInsert 环境并执行稳定的零动作控制，
# 让机械臂初始化到当前姿态并保持静止（不抽搐、不旋转）

# ./isaaclab.sh -p custom/record_peginsert_demo.py \
#   --task Isaac-Factory-PegInsert-Direct-v0 \
#   --num_envs 1 --device cuda

import argparse
import sys
import time
from distutils.util import strtobool
import torch

from isaaclab.app import AppLauncher

# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------
parser = argparse.ArgumentParser(description="Zero-action rollout for Factory PegInsert task.")
parser.add_argument("--task", type=str, default="Isaac-Factory-PegInsert-Direct-v0")
parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=500)
parser.add_argument("--sleep", type=float, default=0.02)
# ⚠️ 不添加 --device；AppLauncher 会自动添加

# 添加 AppLauncher 参数
AppLauncher.add_app_launcher_args(parser)

# 解析 CLI
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

# 启动 Omniverse App
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -------------------------------------------------------------
# 其余导入
# -------------------------------------------------------------
import gymnasium as gym
import omni
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg
from isaaclab_rl.rl_games import RlGamesVecEnvWrapper
import isaaclab_tasks  # noqa: F401


# -------------------------------------------------------------
# 辅助函数：兼容不同 Gym API
# -------------------------------------------------------------
def step_compat(env, action):
    """兼容 Gymnasium / IsaacLab step 返回值"""
    try:
        return env.step(action)
    except ValueError:
        obs, rew, done, info = env.step(action)
        term, trunc = done, done
        return obs, rew, term, trunc, info


# -------------------------------------------------------------
# 稳定初始化函数
# -------------------------------------------------------------
def stabilize_zero_action(env, device):
    """让当前机械臂姿态成为 zero-action 的稳态位姿"""
    try:
        robot = env.unwrapped.scene["robot"]

        # 1️⃣ 让物理系统稳定几步
        for _ in range(5):
            env.unwrapped.sim.step(render=False)

        # 2️⃣ 读取当前关节角与速度
        qpos = robot.data.joint_pos.clone()
        qvel = torch.zeros_like(robot.data.joint_vel)

        # 3️⃣ 写回并固定当前状态
        robot.write_joint_state(qpos, qvel)
        env.unwrapped.sim.step(render=False)

        # 4️⃣ 更新控制器默认目标，使 zero-action = 当前姿态
        if hasattr(env.unwrapped, "controllers") and "arm_action" in env.unwrapped.controllers:
            ctrl = env.unwrapped.controllers["arm_action"]
            if hasattr(ctrl, "set_default_target"):
                ctrl.set_default_target(qpos)
                print("[INIT] Controller default target set to current joint pose.")

        # 5️⃣ 清空动作缓存
        if hasattr(env.unwrapped, "managers") and "action_manager" in env.unwrapped.managers:
            amgr = env.unwrapped.managers["action_manager"]
            if hasattr(amgr, "reset"):
                amgr.reset()
                print("[INIT] Action manager reset for clean start.")

        print("[INIT] Robot stabilized; zero-action now matches current pose.")

    except Exception as e:
        print("[WARN] Failed to align zero-action with current pose:", e)


# -------------------------------------------------------------
# 主函数（Hydra 装饰器）
# -------------------------------------------------------------
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: dict):
    print(f"[INFO] Launching {args_cli.task} with Hydra.")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    # =========================================================
    # ⚙️ 稳定性防抽搐补丁
    # =========================================================
    try:
        env_cfg.actions.arm_action.use_default_offset = False
        print("[PATCH] use_default_offset=False")
    except Exception:
        pass

    try:
        env_cfg.actions.arm_action.scale = 0.2
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
        print("[PATCH] Increased PhysX stability parameters.")
    except Exception:
        pass

    LOCK_WRIST_ROT_IN_LOOP = True

    # =========================================================
    # 🧩 创建环境（与 train.py 一致）
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

    # =========================================================
    # 🦾 初始化机械臂到稳定姿态
    # =========================================================
    stabilize_zero_action(env, args_cli.device)

    print("[INFO] Starting zero-action rollout (Ctrl+C to exit).")
    step_i = 0
    while simulation_app.is_running() and step_i < args_cli.max_steps:
        action = torch.zeros((env.unwrapped.num_envs,
                              env.unwrapped.action_space.shape[-1]),
                             device=args_cli.device)
        

        # 锁 wrist 旋转（避免高速抽搐）
        #if LOCK_WRIST_ROT_IN_LOOP and action.shape[-1] >= 6:
            #action[..., 3:6] = 0.0

        action = torch.clamp(action, -1.0, 1.0)
        obs, rew, term, info = step_compat(env, action)

        if step_i % 10 == 0:
            mean_r = torch.mean(torch.tensor(rew)).item()
            print(f"[Step {step_i}] mean_reward={mean_r:.4f}")

        step_i += 1
        time.sleep(args_cli.sleep)

    env.close()
    print("[INFO] Rollout finished.")


if __name__ == "__main__":
    main()
    simulation_app.close()
