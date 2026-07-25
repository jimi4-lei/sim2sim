from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.core.velocity.config.dog.flat_env_cfg import DogFlatEnvCfg

# 检查 URDF 文件是否存在
urdf_path = "/home/jimi/IsaacLab/source/isaaclab_assets/data/robots/dog/dog.urdf"
print(f"Checking URDF: {urdf_path}")
print(f"File exists: {os.path.exists(urdf_path)}")

if os.path.exists(urdf_path):
    with open(urdf_path, 'r') as f:
        content = f.read()
        print(f"URDF size: {len(content)} bytes")
        print("First 500 chars:")
        print(content[:500])

try:
    env_cfg = DogFlatEnvCfg()
    env_cfg.scene.num_envs = 1
    print("Creating environment...")
    env = ManagerBasedRLEnv(cfg=env_cfg)
    print("Environment created successfully!")
    obs_dict, _ = env.reset()
    print("Reset successful!")
    print("Observation shape:", obs_dict['policy'].shape)
    env.close()
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

simulation_app.close()
