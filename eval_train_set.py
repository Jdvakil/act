"""Check deployed ACT/PACT action predictions on demonstrations without simulation.

Chunk length comes from checkpoint weights; camera order comes from convert_meta.json
or --camera_names. --split train/val reconstructs the current loader's seed-1 80/20
split. This is a legacy reconstruction, not an independently held-out test.
Use --episode_ids for an explicit JSON list of converted episode IDs.
Low action error does not establish rollout success or identify its failure cause.
The existing ACT model constructor requires CUDA even with --device cpu.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import EpisodicDataset, set_seed


def select_episode_ids(num_episodes, split, split_seed=1, explicit_ids=None, limit=None):
    if num_episodes < 1:
        raise ValueError("num_episodes must be positive")
    if explicit_ids is not None:
        ids = np.asarray(explicit_ids)
        if ids.ndim != 1 or ids.dtype.kind not in "iu":
            raise ValueError("episode_ids must be a nonempty JSON list of integers")
        if len(set(ids.tolist())) != len(ids) or np.any(ids < 0) or np.any(ids >= num_episodes):
            raise ValueError("episode_ids must be unique and within the dataset")
    elif split == "all":
        ids = np.arange(num_episodes)
    elif split in ("train", "val"):
        ids = np.random.RandomState(split_seed).permutation(num_episodes)
        cut = int(0.8 * num_episodes)
        ids = ids[:cut] if split == "train" else ids[cut:]
    else:
        raise ValueError(f"unknown split: {split}")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit_episodes must be positive")
        ids = ids[:limit]
    if not len(ids):
        raise ValueError("selected episode partition is empty")
    return ids


def resolve_cameras(data_dir, camera_names=None):
    if camera_names:
        return list(camera_names)
    meta_path = data_dir / "convert_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    cameras = meta.get("camera_names")
    if not cameras:
        raise ValueError("camera order missing; pass --camera_names in training order")
    return list(cameras)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt_dir", required=True, type=Path)
    p.add_argument("--ckpt_name", default="policy_best.ckpt")
    p.add_argument("--data_dir", required=True, type=Path)
    p.add_argument("--num_episodes", type=int, help="Total dataset size, before splitting")
    p.add_argument("--chunk", type=int, help="Optional assertion; must match checkpoint")
    p.add_argument("--camera_names", nargs="+")
    p.add_argument("--split", choices=("train", "val", "all"), default="train")
    p.add_argument("--split_seed", type=int, default=1,
                   help="Current training loader uses seed 1, independently of model seed")
    p.add_argument("--episode_ids", type=Path, help="JSON list of explicit episode IDs")
    p.add_argument("--limit_episodes", type=int, help="Cap selected episodes after splitting")
    p.add_argument("--passes", type=int, default=5, help="Random start frames per episode")
    p.add_argument("--output", type=Path, help="Write metrics and selected IDs as JSON")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.passes < 1:
        p.error("passes must be positive")
    set_seed(0)
    meta_path = args.data_dir / "convert_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    num_episodes = args.num_episodes if args.num_episodes is not None else meta.get("num_episodes")
    if num_episodes is None:
        p.error("pass --num_episodes (total dataset size) or provide convert_meta.json")
    try:
        cameras = resolve_cameras(args.data_dir, args.camera_names)
        episode_ids = select_episode_ids(
            num_episodes, args.split, args.split_seed,
            json.loads(args.episode_ids.read_text()) if args.episode_ids else None,
            args.limit_episodes,
        )
    except ValueError as error:
        p.error(str(error))

    from attn_heatmap import build_policy
    from encoders.pact import encode_for_act

    policy, stats, extractor, _, label = build_policy(
        args.ckpt_dir, args.device, camera_names=cameras, ckpt_name=args.ckpt_name,
    )
    chunk = int(policy.model.query_embed.num_embeddings)
    if args.chunk is not None and args.chunk != chunk:
        p.error(f"--chunk {args.chunk} differs from checkpoint chunk {chunk}")
    is_pact = extractor is not None
    pcfg_path = args.ckpt_dir / "prox_config.json"
    pcfg = json.loads(pcfg_path.read_text()) if pcfg_path.is_file() else {}
    ds = EpisodicDataset(
        episode_ids, str(args.data_dir), cameras, stats, chunk,
        load_proximity=is_pact,
        proximity_layout=pcfg.get("proximity_layout", "raw"),
        n_proximity_sensors=int(pcfg.get("n_proximity_sensors") or 0),
        proximity_feature_dim=int(pcfg.get("prox_feat_dim") or 3),
    )
    # A single worker makes fixed-seed start-frame sampling reproducible and keeps
    # the small diagnostic from spawning extra processes or rendering anything.
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    action_std = torch.as_tensor(stats["action_std"], dtype=torch.float32, device=args.device)
    sum_l1_norm = sum_arm_rad = sum_gripper_norm = 0.0
    n_elements = n_arm = n_gripper = 0
    with torch.inference_mode():
        for _ in range(args.passes):
            for batch in dl:
                if is_pact:
                    image, qpos, action, is_pad, prox = batch
                    prox_pos = encode_for_act(extractor, prox.to(args.device))
                else:
                    image, qpos, action, is_pad = batch
                    prox_pos = None
                action = action.to(args.device)
                a_hat = policy(qpos.to(args.device), image.to(args.device),
                               proximity_positions=prox_pos)
                mask = (~is_pad.to(args.device)).unsqueeze(-1).float()
                err = (a_hat - action).abs()
                valid = int(mask.sum())
                sum_l1_norm += float((err * mask).sum())
                n_elements += valid * action.shape[-1]
                sum_arm_rad += float((err[..., :7] * action_std[:7] * mask).sum())
                n_arm += valid * 7
                sum_gripper_norm += float((err[..., 7:] * mask).sum())
                n_gripper += valid * (action.shape[-1] - 7)
    selected_split = "explicit" if args.episode_ids else args.split
    result = {
        "checkpoint": str((args.ckpt_dir / args.ckpt_name).resolve()),
        "dataset": str(args.data_dir.resolve()), "split": selected_split,
        "split_seed": args.split_seed, "episode_ids": episode_ids.tolist(),
        "num_episodes_total": num_episodes, "passes": args.passes,
        "chunk_size": chunk, "camera_names": cameras,
        "normalized_action_l1": sum_l1_norm / max(n_elements, 1),
        "arm_joint_mae_rad": sum_arm_rad / max(n_arm, 1),
        "gripper_normalized_l1": sum_gripper_norm / max(n_gripper, 1),
        "valid_predicted_steps": n_arm // 7,
        "note": "Offline z=0 action error; not rollout success. Check split and normalization provenance.",
    }
    print(f"\nOffline {selected_split} action check: {label} ({args.ckpt_dir.name})")
    print(f"  episodes={len(episode_ids)} chunk={chunk} cameras={cameras}")
    print(f"  normalized action L1  : {result['normalized_action_l1']:.4f}")
    print(f"  arm joint MAE (rad)   : {result['arm_joint_mae_rad']:.4f}")
    print(f"  normalized gripper L1: {result['gripper_normalized_l1']:.4f}")
    print("  Action prediction only; legacy train/val splits can share scenes and normalization stats.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"  wrote {args.output}")


if __name__ == "__main__":
    main()
