import json
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt


DATA_DIR = Path("/home/qinzhengfangli/molmo_act_data/raw_h5")
OUT_DIR = Path("/home/qinzhengfangli/act/train_val_debug")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 0
TRAIN_RATIO = 0.8


def decode_json_row(row):
    b = bytes(row.tolist())
    s = b.split(b"\x00", 1)[0].decode("utf-8")
    return json.loads(s)


def qpos_to_vec(x):
    arm = x.get("arm", [])
    gripper = x.get("gripper", [])
    vec = list(arm) + list(gripper)
    if len(vec) < 9:
        vec += [0.0] * (9 - len(vec))
    return np.asarray(vec[:9], dtype=np.float32)


def action_to_vec(x):
    arm = x.get("arm", [])
    gripper = x.get("gripper", [])

    if len(gripper) == 0:
        g = 0.0
    else:
        g = float(gripper[0])

    if g > 1.0:
        g = (g / 255.0) * 0.824

    gripper = [g, g]
    vec = list(arm) + list(gripper)
    if len(vec) < 9:
        vec += [0.0] * (9 - len(vec))
    return np.asarray(vec[:9], dtype=np.float32)


def load_episode(path):
    with h5py.File(path, "r") as f:
        traj = f["traj_0"]
        T = traj["obs/agent/qpos"].shape[0]

        qpos = []
        action = []
        rewards = np.asarray(traj["rewards"][:]) if "rewards" in traj else None
        success = np.asarray(traj["success"][:]) if "success" in traj else None

        for t in range(T):
            q = decode_json_row(traj["obs/agent/qpos"][t])
            a = decode_json_row(traj["actions/joint_pos"][t])
            qpos.append(qpos_to_vec(q))
            action.append(action_to_vec(a))

        qpos = np.stack(qpos)
        action = np.stack(action)

    return {
        "T": T,
        "qpos": qpos,
        "action": action,
        "delta": action - qpos,
        "rewards": rewards,
        "success": success,
    }


def summarize_split(name, paths):
    all_qpos = []
    all_action = []
    all_delta = []
    lengths = []
    success_any = []

    for p in paths:
        ep = load_episode(p)
        lengths.append(ep["T"])
        all_qpos.append(ep["qpos"])
        all_action.append(ep["action"])
        all_delta.append(ep["delta"])
        if ep["success"] is not None:
            success_any.append(bool(np.any(ep["success"])))

    qpos = np.concatenate(all_qpos, axis=0)
    action = np.concatenate(all_action, axis=0)
    delta = np.concatenate(all_delta, axis=0)

    summary = {
        "split": name,
        "num_episodes": len(paths),
        "num_timesteps": int(qpos.shape[0]),
        "episode_len_min": int(np.min(lengths)),
        "episode_len_mean": float(np.mean(lengths)),
        "episode_len_max": int(np.max(lengths)),
        "success_episode_count": int(np.sum(success_any)) if success_any else None,
        "qpos_mean": qpos.mean(axis=0).tolist(),
        "qpos_std": qpos.std(axis=0).tolist(),
        "qpos_min": qpos.min(axis=0).tolist(),
        "qpos_max": qpos.max(axis=0).tolist(),
        "action_mean": action.mean(axis=0).tolist(),
        "action_std": action.std(axis=0).tolist(),
        "action_min": action.min(axis=0).tolist(),
        "action_max": action.max(axis=0).tolist(),
        "delta_mean": delta.mean(axis=0).tolist(),
        "delta_std": delta.std(axis=0).tolist(),
        "delta_min": delta.min(axis=0).tolist(),
        "delta_max": delta.max(axis=0).tolist(),
        "mean_abs_action_minus_qpos": float(np.mean(np.abs(delta))),
    }

    return summary, qpos, action, delta


def plot_hist(train_arr, val_arr, title, filename):
    plt.figure()
    plt.hist(train_arr.reshape(-1), bins=80, alpha=0.5, label="train")
    plt.hist(val_arr.reshape(-1), bins=80, alpha=0.5, label="val")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename)
    plt.close()


def plot_dim_means(train_arr, val_arr, title, filename):
    plt.figure()
    x = np.arange(train_arr.shape[1])
    plt.plot(x, train_arr.mean(axis=0), marker="o", label="train mean")
    plt.plot(x, val_arr.mean(axis=0), marker="o", label="val mean")
    plt.legend()
    plt.title(title)
    plt.xlabel("dimension")
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename)
    plt.close()


def main():
    files = sorted(DATA_DIR.glob("*.h5"))
    print("num files:", len(files))

    rng = np.random.default_rng(SEED)
    indices = np.arange(len(files))
    rng.shuffle(indices)

    split = int(TRAIN_RATIO * len(files))
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_paths = [files[i] for i in train_idx]
    val_paths = [files[i] for i in val_idx]

    split_info = {
        "seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "num_total": len(files),
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
        "train_files": [str(p) for p in train_paths],
        "val_files": [str(p) for p in val_paths],
    }
    (OUT_DIR / "split_info.json").write_text(json.dumps(split_info, indent=2))

    train_summary, train_qpos, train_action, train_delta = summarize_split("train", train_paths)
    val_summary, val_qpos, val_action, val_delta = summarize_split("val", val_paths)

    summary = {
        "train": train_summary,
        "val": val_summary,
    }
    (OUT_DIR / "train_val_data_summary.json").write_text(json.dumps(summary, indent=2))

    plot_hist(train_action, val_action, "Action distribution: train vs val", "hist_action_train_val.png")
    plot_hist(train_qpos, val_qpos, "Qpos distribution: train vs val", "hist_qpos_train_val.png")
    plot_hist(train_delta, val_delta, "Action - qpos distribution: train vs val", "hist_delta_train_val.png")

    plot_dim_means(train_action, val_action, "Action mean by dim", "mean_action_by_dim.png")
    plot_dim_means(train_qpos, val_qpos, "Qpos mean by dim", "mean_qpos_by_dim.png")
    plot_dim_means(train_delta, val_delta, "Action - qpos mean by dim", "mean_delta_by_dim.png")

    print(json.dumps(summary, indent=2))
    print("saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
