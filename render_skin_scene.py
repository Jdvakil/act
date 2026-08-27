from __future__ import annotations

print('[debug] script started', flush=True)

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.pop("DISPLAY", None)

# paths
ACT_ROOT = Path("/home/qinzhengfangli/act")
MOLMO_ROOT = Path("/home/qinzhengfangli/molmospaces_jay")

sys.path.insert(0, str(ACT_ROOT))
sys.path.insert(0, str(MOLMO_ROOT))

print('[debug] before cv2 import', flush=True)
import cv2
print('[debug] after cv2 import', flush=True)
import numpy as np

print('[debug] before molmospaces config import', flush=True)
from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
    FrankaSkinPickAndPlacePilotMediumConfig,
)
print('[debug] after molmospaces config import', flush=True)


def main():
    out_dir = Path("/home/qinzhengfangli/act/skin_scene_images")
    out_dir.mkdir(parents=True, exist_ok=True)

    print('[debug] before cfg init', flush=True)
    cfg = FrankaSkinPickAndPlacePilotMediumConfig()
    print('[debug] after cfg init', flush=True)

    # make sure videos/viewer are off; we only want render screenshots
    cfg.use_passive_viewer = False
    cfg.filter_for_successful_trajectories = False

    # Use one house/task first
    cfg.task_sampler_config.samples_per_house = 1
    cfg.task_sampler_config.max_tasks = 1

    print("[render] config:", cfg)
    print("[render] robot:", cfg.robot_config)
    print("[render] camera_config:", cfg.camera_config)

    sampler = cfg.task_sampler_config.task_sampler_class(cfg)

    print("[render] sampling task...")
    task = sampler.sample_task()
    print("[render] sampled task:", task)

    obs, info = task.reset()
    print("[render] obs keys:", list(obs.keys()))

    # Save all RGB camera observations
    saved = 0
    for k, v in obs.items():
        if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[-1] == 3:
            img = v
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            # OpenCV writes BGR
            out = out_dir / f"{k}.png"
            cv2.imwrite(str(out), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print("[render] saved", out, img.shape)
            saved += 1

    # Step a little so the robot starts operating, using zero/hold action if possible
    # This is mainly to get a non-initial scene screenshot.
    for t in range(5):
        try:
            action = {
                "arm": np.asarray(obs["qpos"]["arm"][:7], dtype=np.float32),
                "gripper": np.asarray([0.0], dtype=np.float32),
            }
            obs, reward, terminated, truncated, info = task.step(action)
        except Exception as e:
            print("[render] step failed, stopping:", repr(e))
            break

    # Save after-step RGB images
    for k, v in obs.items():
        if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[-1] == 3:
            img = v
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            out = out_dir / f"after_step_{k}.png"
            cv2.imwrite(str(out), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print("[render] saved", out, img.shape)
            saved += 1

    print("[render] done, saved images:", saved)
    print("[render] output dir:", out_dir)


if __name__ == "__main__":
    main()
