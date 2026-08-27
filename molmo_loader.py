import os
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch


CAMERA_FILE_MAP = {
    "wrist": "episode_00000000_wrist_camera_zed_mini_batch_1_of_1.mp4",
    "top": "episode_00000000_randomized_zed2_analogue_1_batch_1_of_1.mp4",
    "zed2_1": "episode_00000000_randomized_zed2_analogue_1_batch_1_of_1.mp4",
    "zed2_2": "episode_00000000_randomized_zed2_analogue_2_batch_1_of_1.mp4",
    "gopro": "episode_00000000_randomized_gopro_analogue_1_batch_1_of_1.mp4",
    "shoulder": "episode_00000000_droid_shoulder_light_randomization_batch_1_of_1.mp4",
}


def _decode_json_row(row):
    b = bytes(row.tolist())
    s = b.split(b"\x00", 1)[0].decode("utf-8")
    return json.loads(s)


def _qpos_to_vec(x):
    # qpos has: arm 7 + gripper 2 = 9 dims
    arm = x.get("arm", [])
    gripper = x.get("gripper", [])
    vec = list(arm) + list(gripper)

    if len(vec) < 9:
        vec = vec + [0.0] * (9 - len(vec))
    return np.asarray(vec[:9], dtype=np.float32)


def _action_to_vec(x):
    # action has: arm 7 + gripper 1.
    # In these MolmoSpaces h5 files, gripper action appears encoded as 0/255.
    # Convert it to the same physical scale as qpos gripper, roughly 0.0 to 0.824.
    arm = x.get("arm", [])
    gripper = x.get("gripper", [])

    if len(gripper) == 0:
        g = 0.0
    else:
        g = float(gripper[0])

    # Convert uint8-style gripper command to physical gripper opening scale.
    # If it is already small, leave it alone.
    if g > 1.0:
        g = (g / 255.0) * 0.824

    gripper = [g, g]
    vec = list(arm) + gripper

    if len(vec) < 9:
        vec = vec + [0.0] * (9 - len(vec))
    return np.asarray(vec[:9], dtype=np.float32)


def _read_video_frame(video_path, frame_idx, image_size=64):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
    frame = frame.astype(np.float32) / 255.0
    frame = np.transpose(frame, (2, 0, 1))  # C, H, W
    return frame


class MolmoDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, camera_names, num_queries, image_size=64):
        self.data_dir = Path(data_dir)
        self.camera_names = camera_names
        self.num_queries = num_queries
        self.image_size = image_size

        self.h5_files = sorted(self.data_dir.glob("*.h5"))
        if len(self.h5_files) == 0:
            raise RuntimeError(f"No .h5 files found in {self.data_dir}")

    def __len__(self):
        return len(self.h5_files)

    def __getitem__(self, idx):
        h5_path = self.h5_files[idx]
        real_h5_path = Path(os.path.realpath(h5_path))
        house_dir = real_h5_path.parent

        with h5py.File(real_h5_path, "r") as f:
            traj = f["traj_0"]

            qpos_raw = traj["obs/agent/qpos"]
            action_raw = traj["actions/joint_pos"]
            T = qpos_raw.shape[0]

            start_ts = np.random.randint(0, T)

            qpos = _qpos_to_vec(_decode_json_row(qpos_raw[start_ts]))

            action_seq = []
            for t in range(start_ts, T):
                action_seq.append(_action_to_vec(_decode_json_row(action_raw[t])))
            action_seq = np.stack(action_seq, axis=0)

        action_len = len(action_seq)
        padded_action = np.zeros((self.num_queries, 9), dtype=np.float32)
        is_pad = np.ones((self.num_queries,), dtype=bool)

        n = min(action_len, self.num_queries)
        padded_action[:n] = action_seq[:n]
        is_pad[:n] = False

        images = []
        for cam_name in self.camera_names:
            video_name = CAMERA_FILE_MAP.get(cam_name, CAMERA_FILE_MAP["wrist"])
            video_path = house_dir / video_name
            images.append(_read_video_frame(video_path, start_ts, self.image_size))

        image = np.stack(images, axis=0)  # num_cams, 3, H, W

        return (
            torch.tensor(image).float(),
            torch.tensor(qpos).float(),
            torch.tensor(padded_action).float(),
            torch.tensor(is_pad).bool(),
        )
