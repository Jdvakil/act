"""
ACT policy evaluation in an Isaac Lab environment.

Must be run via the Isaac Lab python launcher:
    python eval_isaac.py --task <TASK_NAME> --ckpt_dir <CKPT_DIR> [AppLauncher args]

Example:
    python eval_isaac.py \
        --task Isaac-Stack-Cube-Franka-IK-Abs-v0 \
        --ckpt_dir /home/jaydv/code/act/checkpoints/isaac_task \
        --num_rollouts 50 \
        --chunk_size 100 \
        --temporal_agg \
        --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate ACT policy in Isaac Lab.")
parser.add_argument("--task", type=str, required=True, help="Isaac Lab task name.")
parser.add_argument("--ckpt_dir", type=str, required=True, help="Directory containing policy_best.ckpt and dataset_stats.pkl.")
parser.add_argument("--ckpt_name", type=str, default="policy_best.ckpt")
parser.add_argument("--num_rollouts", type=int, default=50)
parser.add_argument("--horizon", type=int, default=200, help="Max steps per rollout.")
parser.add_argument("--chunk_size", type=int, default=100, help="ACT chunk size (num_queries).")
parser.add_argument("--hidden_dim", type=int, default=512)
parser.add_argument("--dim_feedforward", type=int, default=3200)
parser.add_argument("--state_dim", type=int, default=7, help="Action dimension.")
parser.add_argument("--qpos_dim", type=int, default=9, help="Joint-position input dimension.")
parser.add_argument(
    "--camera_names", nargs="+", default=["table_cam", "wrist_cam"],
    help="Camera observation keys to feed to the policy.",
)
parser.add_argument("--temporal_agg", action="store_true")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--save_video", action="store_true", help="Save per-rollout MP4 videos to ckpt_dir.")
parser.add_argument("--video_fps", type=int, default=20, help="FPS for saved videos (Isaac step dt=0.05 → 20 fps).")
parser.add_argument("--video_scale", type=int, default=3, help="Integer upscale factor for saved video frames (default 3 → 200px becomes 600px).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import pickle
import random

import gymnasium as gym
import cv2
import numpy as np
import torch
from einops import rearrange
import torchvision.transforms as transforms

import isaaclab_tasks  # noqa: F401
import isaaclab_mimic.envs  # noqa: F401

from isaaclab_tasks.utils import parse_env_cfg

# ── ACT imports ──────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(__file__))
from policy import ACTPolicy
from utils import set_seed
from visualize_episodes import save_videos


# ── helpers ──────────────────────────────────────────────────────────────────

def build_policy(args_cli) -> ACTPolicy:
    policy_config = {
        "lr": 1e-5,
        "num_queries": args_cli.chunk_size,
        "kl_weight": 10,
        "hidden_dim": args_cli.hidden_dim,
        "dim_feedforward": args_cli.dim_feedforward,
        "lr_backbone": 1e-5,
        "backbone": "resnet18",
        "enc_layers": 4,
        "dec_layers": 7,
        "nheads": 8,
        "camera_names": args_cli.camera_names,
        "state_dim": args_cli.state_dim,
        "qpos_dim": args_cli.qpos_dim,
    }
    return ACTPolicy(policy_config)


def get_image(obs_dict: dict, camera_names: list[str]) -> torch.Tensor:
    """Extract and stack camera images from the Isaac Lab obs dict.

    Isaac Lab stores images as (1, H, W, C) uint8 tensors.
    We need (1, num_cams, C, H, W) float32 in [0, 1].
    """
    frames = []
    for cam in camera_names:
        img = obs_dict[cam]  # (1, H, W, C) or (H, W, C)
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        img = np.squeeze(img)           # (H, W, C)
        img = rearrange(img, "h w c -> c h w")
        frames.append(img)
    arr = np.stack(frames, axis=0)     # (num_cams, C, H, W)
    tensor = torch.from_numpy(arr / 255.0).float().cuda().unsqueeze(0)  # (1, num_cams, C, H, W)
    return tensor


def get_qpos(obs_dict: dict) -> np.ndarray:
    """Extract joint positions from the Isaac Lab obs dict."""
    qpos = obs_dict["joint_pos"]
    if isinstance(qpos, torch.Tensor):
        qpos = qpos.cpu().numpy()
    return np.squeeze(qpos)  # (qpos_dim,)


# ── main eval loop ────────────────────────────────────────────────────────────

def main():
    set_seed(args_cli.seed)
    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    # Load policy
    policy = build_policy(args_cli)
    ckpt_path = os.path.join(args_cli.ckpt_dir, args_cli.ckpt_name)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(f"Loaded checkpoint: {ckpt_path}  status={loading_status}")
    policy.cuda()
    policy.eval()

    # Load normalisation stats
    stats_path = os.path.join(args_cli.ckpt_dir, "dataset_stats.pkl")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    pre_process = lambda qpos: (qpos - stats["qpos_mean"]) / stats["qpos_std"]
    post_process = lambda a: a * stats["action_std"] + stats["action_mean"]

    # Create environment
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg)

    query_frequency = args_cli.chunk_size
    num_queries = args_cli.chunk_size
    if args_cli.temporal_agg:
        query_frequency = 1

    max_timesteps = args_cli.horizon
    successes = []
    image_lists = []

    for rollout_id in range(args_cli.num_rollouts):
        obs_dict, _ = env.reset()
        obs = obs_dict["policy"]

        if args_cli.temporal_agg:
            all_time_actions = torch.zeros(
                [max_timesteps, max_timesteps + num_queries, args_cli.state_dim]
            ).cuda()

        image_list = []
        success = False

        with torch.no_grad():
            for t in range(max_timesteps):
                qpos_numpy = get_qpos(obs)
                qpos_norm = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos_norm).float().cuda().unsqueeze(0)  # (1, qpos_dim)

                curr_image = get_image(obs, args_cli.camera_names)
                # store (H, W, C) uint8 per camera for save_videos
                frame = {}
                for cam in args_cli.camera_names:
                    img = obs[cam]
                    if isinstance(img, torch.Tensor):
                        img = img.cpu().numpy()
                    img = np.squeeze(img)  # (H, W, C)
                    if img.dtype != np.uint8:
                        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                    if args_cli.video_scale != 1:
                        h, w = img.shape[:2]
                        img = cv2.resize(img, (w * args_cli.video_scale, h * args_cli.video_scale),
                                         interpolation=cv2.INTER_LINEAR)
                    frame[cam] = img
                image_list.append(frame)

                # Query ACT policy
                if t % query_frequency == 0:
                    all_actions = policy(qpos, curr_image)  # (1, chunk_size, state_dim)

                if args_cli.temporal_agg:
                    all_time_actions[[t], t : t + num_queries] = all_actions
                    actions_for_step = all_time_actions[:, t]
                    populated = torch.all(actions_for_step != 0, axis=1)
                    actions_for_step = actions_for_step[populated]
                    k = 0.01
                    exp_w = np.exp(-k * np.arange(len(actions_for_step)))
                    exp_w = exp_w / exp_w.sum()
                    exp_w = torch.from_numpy(exp_w).cuda().unsqueeze(1)
                    raw_action = (actions_for_step * exp_w).sum(dim=0, keepdim=True)
                else:
                    raw_action = all_actions[:, t % query_frequency]

                action = post_process(raw_action.squeeze(0).cpu().numpy())
                action_tensor = torch.from_numpy(action).float().cuda().unsqueeze(0)  # (1, action_dim)

                obs_dict, _, terminated, truncated, info = env.step(action_tensor)
                obs = obs_dict["policy"]

                # Check success: look for the success termination term in info
                # Isaac Lab puts per-term termination flags in info["log"] or as top-level keys.
                # The "success" TerminationTerm fires only when all cubes are stacked.
                # "cube_N_dropping" and "time_out" are separate terms.
                if "log" in info and "success" in info["log"]:
                    episode_success = bool(info["log"]["success"])
                elif "success" in info:
                    episode_success = bool(info["success"])
                else:
                    episode_success = False  # can't tell — don't mis-report as success

                if episode_success:
                    success = True
                    break

                # Stop stepping if any terminal condition fired (cube drop, success, timeout)
                if terminated or truncated:
                    break

        successes.append(success)
        print(f"Rollout {rollout_id}: {'SUCCESS' if success else 'FAILURE'}")

        if args_cli.save_video and image_list:
            video_path = os.path.join(args_cli.ckpt_dir, f"video_rollout{rollout_id:03d}.mp4")
            save_videos(image_list, dt=1.0 / args_cli.video_fps, video_path=video_path)

    success_rate = np.mean(successes)
    print(f"\n=== Results over {args_cli.num_rollouts} rollouts ===")
    print(f"Success rate: {success_rate * 100:.1f}%  ({sum(successes)}/{args_cli.num_rollouts})")

    # Save result summary
    result_path = os.path.join(args_cli.ckpt_dir, f"result_{args_cli.ckpt_name.split('.')[0]}.txt")
    with open(result_path, "w") as f:
        f.write(f"Success rate: {success_rate * 100:.1f}%  ({sum(successes)}/{args_cli.num_rollouts})\n")
        f.write(repr(successes))
    print(f"Saved results to {result_path}")

    env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    simulation_app.close()
