# ================================================================
# run_factory_zero_action.py
# 以全零动作运行 Factory Peg Insert 环境，用于验证物理震荡
# ================================================================

from isaaclab.app import AppLauncher

# 1️⃣ 启动 SimulationApp
app_launcher = AppLauncher()
simulation_app = app_launcher.app

# 2️⃣ 导入所需模块
import time
import torch
import gymnasium as gym
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# 3️⃣ 环境配置
TASK_NAME = "Isaac-Factory-PegInsert-Direct-v0"
print(f"[INFO] Loading environment: {TASK_NAME}")

env_cfg = parse_env_cfg(TASK_NAME, device="cuda:0", num_envs=1)
env = gym.make(TASK_NAME, cfg=env_cfg).unwrapped
print("[INFO] Environment created successfully.")

# 4️⃣ 初始化环境
env.sim.reset()
env.reset()

# 5️⃣ 准备全零动作张量
zero_action = torch.zeros((env.num_envs, env.action_space.shape[0]), device=env.device)
print(f"[INFO] Zero action shape: {zero_action.shape}")

# 6️⃣ 主循环：持续以零动作步进环境
print("[INFO] Running simulation with zero actions (press Ctrl+C to stop)...")
step_i = 0
try:
    while simulation_app.is_running():
        env.step(zero_action)

        # 每隔 10 步打印一次状态
        if step_i % 10 == 0:
            # 打印末端执行器角速度与位置
            ee_quat = env._robot.data.body_quat_w[:, env.fingertip_body_idx]
            ee_pos = env._robot.data.body_pos_w[:, env.fingertip_body_idx]
            ee_angvel = env._robot.data.body_ang_vel_w[:, env.fingertip_body_idx]
            mean_angvel = ee_angvel.abs().mean().item()
            print(f"[Step {step_i}] Mean |angvel| = {mean_angvel:.4f}, EE pos = {ee_pos[0].cpu().numpy()}")
        
        step_i += 1
        time.sleep(1 / 30.0)  # 保持视觉稳定（30Hz 渲染）
        env.sim.render()

except KeyboardInterrupt:
    print("\n[INFO] Stopping simulation...")

# 7️⃣ 清理
env.close()
simulation_app.close()
print("[INFO] Done.")
