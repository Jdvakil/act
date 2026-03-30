"""
Convert an IsaacLab/robomimic-format HDF5 dataset to the per-episode HDF5 format
expected by the ACT training pipeline.

Source layout (single file):
    data/demo_0/actions          (T, action_dim)
    data/demo_0/obs/joint_pos    (T, 9)
    data/demo_0/obs/joint_vel    (T, 9)
    data/demo_0/obs/table_cam    (T, H, W, 3)   uint8
    data/demo_0/obs/wrist_cam    (T, H, W, 3)   uint8

Output layout (one file per episode):
    episode_N.hdf5
        /observations/qpos              (T, 9)
        /observations/qvel              (T, 9)
        /observations/images/table_cam  (T, H, W, 3)  uint8
        /observations/images/wrist_cam  (T, H, W, 3)  uint8
        /action                         (T, action_dim)
    attrs: sim = True
"""

import argparse
import os

import h5py
import numpy as np


def convert(input_file, output_dir, camera_names):
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(input_file, "r") as src:
        demo_keys = sorted(src["data"].keys(), key=lambda k: int(k.split("_")[1]))
        print(f"Found {len(demo_keys)} demos in {input_file}")

        for ep_idx, demo_key in enumerate(demo_keys):
            demo = src[f"data/{demo_key}"]

            actions = demo["actions"][()]           # (T, action_dim)
            joint_pos = demo["obs/joint_pos"][()]   # (T, 9)
            joint_vel = demo["obs/joint_vel"][()]   # (T, 9)

            out_path = os.path.join(output_dir, f"episode_{ep_idx}.hdf5")
            with h5py.File(out_path, "w") as dst:
                dst.attrs["sim"] = True

                obs_grp = dst.create_group("observations")
                obs_grp.create_dataset("qpos", data=joint_pos.astype(np.float32))
                obs_grp.create_dataset("qvel", data=joint_vel.astype(np.float32))

                img_grp = obs_grp.create_group("images")
                for cam in camera_names:
                    cam_data = demo[f"obs/{cam}"][()]   # (T, H, W, 3) uint8
                    img_grp.create_dataset(cam, data=cam_data)

                dst.create_dataset("action", data=actions.astype(np.float32))

            if (ep_idx + 1) % 50 == 0 or ep_idx == len(demo_keys) - 1:
                print(f"  Wrote {ep_idx + 1}/{len(demo_keys)} episodes")

    print(f"\nDone. {len(demo_keys)} episodes saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, help="Path to the Isaac HDF5 dataset")
    parser.add_argument("--output_dir", required=True, help="Directory to write per-episode HDF5 files")
    parser.add_argument(
        "--camera_names",
        nargs="+",
        default=["table_cam", "wrist_cam"],
        help="Camera keys to include (default: table_cam wrist_cam)",
    )
    args = parser.parse_args()
    convert(args.input_file, args.output_dir, args.camera_names)
