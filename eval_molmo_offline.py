import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
import numpy as np

from molmo_loader import MolmoDataset
from imitate_episodes import make_policy


def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = MolmoDataset(
        data_dir=args.data_dir,
        camera_names=args.camera_names.split(","),
        num_queries=args.chunk_size,
    )

    # deterministic 80/20 split, similar to training/validation setup
    n = len(dataset)
    indices = np.arange(n)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(indices)

    split = int(0.8 * n)
    train_indices = indices[:split]
    val_indices = indices[split:]

    if args.split == "train":
        eval_dataset = Subset(dataset, train_indices)
    elif args.split == "val":
        eval_dataset = Subset(dataset, val_indices)
    elif args.split == "all":
        eval_dataset = dataset
    else:
        raise ValueError(f"Unknown split: {args.split}")

    loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    policy_config = {
        "lr": args.lr,
        "num_queries": args.chunk_size,
        "kl_weight": args.kl_weight,
        "hidden_dim": args.hidden_dim,
        "dim_feedforward": args.dim_feedforward,
        "lr_backbone": args.lr_backbone,
        "backbone": "resnet18",
        "enc_layers": 4,
        "dec_layers": 7,
        "nheads": 8,
        "camera_names": args.camera_names.split(","),
        "state_dim": args.state_dim,
    }

    policy = make_policy("ACT", policy_config)

    ckpt = torch.load(args.ckpt, map_location=device)
    policy.load_state_dict(ckpt)
    policy.to(device)
    policy.eval()

    metrics_sum = {}
    count = 0

    with torch.inference_mode():
        for batch in loader:
            image, qpos, action, is_pad = batch
            image = image.to(device)
            qpos = qpos.to(device)
            action = action.to(device)
            is_pad = is_pad.to(device)

            out = policy(qpos, image, action, is_pad)

            bs = image.shape[0]
            count += bs

            for k, v in out.items():
                if torch.is_tensor(v):
                    metrics_sum[k] = metrics_sum.get(k, 0.0) + float(v.detach().cpu()) * bs

    metrics = {k: v / count for k, v in metrics_sum.items()}

    result = {
        "checkpoint": args.ckpt,
        "data_dir": args.data_dir,
        "split": args.split,
        "num_total_episodes": n,
        "num_eval_episodes": len(eval_dataset),
        "camera_names": args.camera_names.split(","),
        "chunk_size": args.chunk_size,
        "batch_size": args.batch_size,
        "state_dim": args.state_dim,
        "metrics": metrics,
    }

    print(json.dumps(result, indent=2))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"Saved results to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_dir", default="/home/qinzhengfangli/molmo_act_data/raw_h5")
    parser.add_argument("--split", default="val", choices=["train", "val", "all"])
    parser.add_argument("--camera_names", default="wrist")
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--state_dim", type=int, default=9)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--kl_weight", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dim_feedforward", type=int, default=3200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    evaluate(args)
