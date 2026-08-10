"""Evaluate the DEPLOYED policy (z=0, inference mode) on the TRAINING set — open-loop
action-prediction error vs the expert demos. Distinguishes the failure mode behind the
poor closed-loop rollout success:
  * train L1 high              -> underfit (didn't learn the demos)
  * train L1 low  ~ val L1 low -> fits demos fine; rollout fails from COVARIATE SHIFT
                                  (BC drifts into states the demos never cover)
  * train L1 << val L1         -> overfit / memorization
No sim needed — replays the converted training episodes through the policy.

Run (from submodules/act, mlspaces env):
    python eval_train_set.py --ckpt_dir ckpts/obstacle_pact/<dated>_pact_delta \
        --data_dir /home/jaydv/code/prox_learning/act_style_data/obstacle_prox_v1 \
        --num_episodes 100 --passes 5
Proximity is auto-detected from the ckpt's prox_config.json (same as eval).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from attn_heatmap import build_policy   # builds ACTPolicy + stats + prox extractor (no sim)
from utils import EpisodicDataset, set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True, type=Path)
    p.add_argument("--data_dir", required=True, type=Path)
    p.add_argument("--num_episodes", type=int, default=100)
    p.add_argument("--chunk", type=int, default=100)
    p.add_argument("--passes", type=int, default=5, help="passes over the data (random start_ts each)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    set_seed(0)

    policy, stats, extractor, n_prox, label = build_policy(args.ckpt_dir, args.device)
    is_pact = extractor is not None

    ds = EpisodicDataset(np.arange(args.num_episodes), str(args.data_dir),
                         ["exo_camera_1", "wrist_camera"], stats, args.chunk,
                         load_proximity=is_pact)
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=2)
    action_std = torch.tensor(np.asarray(stats["action_std"]), dtype=torch.float32, device=args.device)

    tot_n = 0          # masked element count
    sum_l1_norm = 0.0  # normalized-space |err|, summed over valid elements (all 8 dims)
    sum_arm_rad = 0.0  # de-normalized arm-joint |err| (rad), summed over valid (7 arm dims)
    n_arm = 0
    with torch.inference_mode():
        for _ in range(args.passes):
            for batch in dl:
                if is_pact:
                    image, qpos, action, is_pad, prox = batch
                    prox = prox.to(args.device)
                    prox_pos = extractor(prox)
                else:
                    image, qpos, action, is_pad = batch
                    prox_pos = None
                image = image.to(args.device); qpos = qpos.to(args.device)
                action = action.to(args.device); is_pad = is_pad.to(args.device)
                a_hat = policy(qpos, image, proximity_positions=prox_pos)   # (B, chunk, 8) z=0
                mask = (~is_pad).unsqueeze(-1).float()                      # (B, chunk, 1)
                err = (a_hat - action).abs()                               # normalized space
                sum_l1_norm += float((err * mask).sum())
                tot_n += float(mask.sum()) * action.shape[-1]
                # arm joints (dims 0:7) in radians: de-normalize by action_std
                arm_err = err[..., :7] * action_std[:7]
                sum_arm_rad += float((arm_err * mask).sum())
                n_arm += float(mask.sum()) * 7

    l1_norm = sum_l1_norm / max(tot_n, 1)
    arm_rad = sum_arm_rad / max(n_arm, 1)
    print(f"\n==== TRAIN-SET open-loop eval: {label} ({args.ckpt_dir.name}) ====")
    print(f"  samples ~= {int(tot_n/ action.shape[-1])} (masked steps), passes={args.passes}")
    print(f"  normalized action L1      : {l1_norm:.4f}   (compare to val loss ~0.078)")
    print(f"  arm-joint mean |err|      : {arm_rad:.4f} rad  ({np.degrees(arm_rad):.2f} deg)")


if __name__ == "__main__":
    main()
