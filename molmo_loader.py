
import torch
import h5py
import cv2
import numpy as np
import os

class MolmoDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, camera_names, num_queries):
        self.data_dir = data_dir
        self.camera_names = camera_names
        self.num_queries = num_queries
        
        self.h5_files = [f for f in os.listdir(data_dir) if f.endswith(".h5")]

    def __len__(self):
        return len(self.h5_files)

    def __getitem__(self, idx):
        h5_path = os.path.join(self.data_dir, self.h5_files[idx])
        f = h5py.File(h5_path, "r")

        traj = f["traj_0"]

        qpos_all = traj["obs/agent/qpos"][:, :9]        # (T, D)
        actions_all = traj["actions/joint_pos"][:, :9]  # (T, D)

        T = len(qpos_all)
        start_ts = np.random.randint(0, T)

        # === 当前 state ===
        qpos = qpos_all[start_ts]

        # === action sequence ===
        action = actions_all[start_ts:]
        action_len = len(action)

        padded_action = np.zeros((self.num_queries, action.shape[1]))
        padded_action[:action_len] = action[:self.num_queries]

        is_pad = np.zeros(self.num_queries)
        is_pad[action_len:] = 1

        image = np.zeros((len(self.camera_names), 3, 64, 64))

        return (
            torch.tensor(image).float(),
            torch.tensor(qpos).float(),
            torch.tensor(padded_action).float(),
            torch.tensor(is_pad).bool()
        )

