from isaaclab_tasks.direct.factory.factory_env_cfg import FactoryTaskPegInsertCfg
from isaaclab_tasks.direct.factory.factory_env import FactoryEnv

cfg = FactoryTaskPegInsertCfg()
env = FactoryEnv(cfg)

obs, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
