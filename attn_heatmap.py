"""Offline ACT attention-saliency heatmaps for FAILURE episodes ("where the policy looks").

Replays saved eval rollouts through the trained policy and overlays the ACT decoder's
cross-attention over the camera tokens onto the exo/wrist frames — showing where the
policy was attending. Identifies failures from the per-house h5 (success[-1]==False).
No molmospaces / no sim needed (pure replay), so it runs while evals are going.

How attention is read (verified): the decoder's last-layer cross-attention
(`transformer.decoder.layers[-1].multihead_attn`, an nn.MultiheadAttention) returns
head-averaged weights (1, num_queries=100, num_keys) via a forward hook. Encoder memory
tokens are [latent, proprio, prox(N*K, PACT only), image(H*W*ncam)], so the image keys
are the trailing H*W*ncam = 8*10*2 = 160 tokens (ResNet18 on 240x320 -> 8x10 grid; two
cameras concatenated along WIDTH -> 8x20). Average over the 100 action queries, slice the
trailing image tokens, reshape (8, 20), split into exo(8,10)+wrist(8,10), upsample to
320x240, overlay.

Usage (from submodules/act, mlspaces env):
    python attn_heatmap.py --ckpt_dir ckpts/obstacle_pact/<dated>_pact_delta \
        --eval_dir /home/jaydv/code/prox_learning/eval_output/eq50_delta \
        --out /home/jaydv/code/prox_learning/eval_output/heatmaps/delta --max_episodes 8
Proximity (PACT) is auto-detected from the ckpt's prox_config.json.
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

from policy import ACTPolicy

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from encoders.pact import (
    build_pact_encoder,
    is_geometry_feature,
    resolve_act_encoder_load,
)

CAM_NAMES = ("exo_camera_1", "wrist_camera")
H_FEAT, W_FEAT = 8, 10          # resnet18 layer4 feature grid for a 240x320 image
IMG_W, IMG_H = 320, 240         # policy input size (eval resizes to this)


@contextmanager
def _detr_argv(ckpt_dir: str):
    """Shield detr/main.py's argparse from this script's CLI flags during model build."""
    orig = sys.argv
    sys.argv = [orig[0] if orig else "attn_heatmap.py",
                "--ckpt_dir", str(ckpt_dir), "--policy_class", "ACT",
                "--task_name", "obstacle_pact", "--seed", "0", "--num_epochs", "1"]
    try:
        yield
    finally:
        sys.argv = orig


def build_policy(ckpt_dir: Path, device: str = "cuda", *, camera_names=None,
                 ckpt_name: str = "policy_best.ckpt"):
    """Rebuild the trained ACTPolicy exactly as eval does (proximity iff prox_config.json)."""
    prox_path = ckpt_dir / "prox_config.json"
    pcfg = json.loads(prox_path.read_text()) if prox_path.exists() else None
    state = torch.load(ckpt_dir / ckpt_name, map_location="cpu", weights_only=True)
    chunk_size = int(state["model.query_embed.weight"].shape[0])
    cfg = dict(lr=1e-5, num_queries=chunk_size, kl_weight=10, hidden_dim=512,
               dim_feedforward=3200, lr_backbone=1e-5, backbone="resnet18",
               enc_layers=4, dec_layers=7, nheads=8,
               camera_names=list(CAM_NAMES if camera_names is None else camera_names),
               state_dim=9, action_dim=8)
    extractor = None
    if pcfg:
        k = int(pcfg.get("prox_tokens_per_sensor", 8))
        feat = pcfg.get("prox_feature", "raw")
        if is_geometry_feature(feat) and k == 8:
            k = 1
        extractor = build_pact_encoder(
            feat,
            device=device,
            layout=pcfg.get("prox_layout", "global"),
            tokens_per_sensor=k,
            **resolve_act_encoder_load(ckpt_dir, pcfg, policy_name=ckpt_name),
        )
        if extractor is not None:
            extractor.eval()
        cfg["n_proximity_sensors"] = extractor.n_act_sensors
        cfg["prox_tokens_per_sensor"] = k
        cfg["prox_feat_dim"] = extractor.act_feat_dim
    with _detr_argv(ckpt_dir):
        policy = ACTPolicy(cfg)
    policy.load_state_dict(state)
    policy.to(device).eval()
    stats = pickle.load(open(ckpt_dir / "dataset_stats.pkl", "rb"))
    n_prox = (cfg.get("n_proximity_sensors", 0) or 0) * (cfg.get("prox_tokens_per_sensor", 1) or 1)
    label = f"PACT-{pcfg['prox_feature']}" if pcfg else "vanilla"
    print(f"[heatmap] loaded {label} from {ckpt_dir.name} | n_prox_tokens={n_prox}")
    return policy, stats, extractor, n_prox, label


def _decode_qpos(blob: np.ndarray) -> np.ndarray:
    raw = bytes(blob).split(b"\x00", 1)[0]
    d = json.loads(raw.decode("utf-8")) if raw else {}
    arm = (d.get("arm") or [])[:7]
    grip = (d.get("gripper") or [])[:2]
    out = np.zeros(9, np.float32)
    out[:len(arm)] = arm
    out[7:7 + len(grip)] = grip
    return out


def _stack_prox(traj: h5py.Group, t: int, sensor_order: list[str]) -> np.ndarray:
    """(40, 8, 8) raw meters at step t, substeps mean-pooled (matches the live eval path)."""
    frames = []
    for name in sensor_order:
        d = np.asarray(traj[f"obs/proximity/{name}"][t], np.float32)  # (n_substeps, 8, 8)
        frames.append(d.mean(0) if d.ndim == 3 else d)
    return np.stack(frames, 0)


def _read_mp4(path: Path) -> list[np.ndarray]:
    """All frames as BGR uint8 resized to (IMG_H, IMG_W)."""
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if fr.shape[:2] != (IMG_H, IMG_W):
            fr = cv2.resize(fr, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        out.append(fr)
    cap.release()
    return out


def _overlay(attn: torch.Tensor, n_prox: int, frames_bgr: dict) -> dict:
    """attn (1,100,num_keys) head-averaged -> per-camera heatmap overlay (BGR)."""
    a = attn[0].mean(0)                                   # (num_keys,) avg over action queries
    start = 2 + n_prox
    img = a[start:start + H_FEAT * W_FEAT * len(CAM_NAMES)].reshape(H_FEAT, len(CAM_NAMES) * W_FEAT)
    img = img.float().cpu().numpy()
    out = {}
    for ci, cam in enumerate(CAM_NAMES):
        g = img[:, ci * W_FEAT:(ci + 1) * W_FEAT]
        g = (g - g.min()) / (np.ptp(g) + 1e-8)
        hm = cv2.resize(g, (IMG_W, IMG_H), interpolation=cv2.INTER_CUBIC)
        hmc = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        out[cam] = cv2.addWeighted(frames_bgr[cam], 0.55, hmc, 0.45, 0)
    return out


def process_episode(policy, stats, extractor, n_prox, traj, mp4s, device, out_path, label, ep_idx):
    """Replay one episode, write an annotated exo|wrist attention MP4. Returns (T, collided)."""
    cap = {}

    def hook(m, i, o):
        cap["attn"] = o[1].detach()
    handle = policy.model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(hook)

    sensor_order = extractor.sensor_order if extractor is not None else None
    qpos_raw = traj["obs/agent/qpos"]
    T = min(len(mp4s[CAM_NAMES[0]]), len(mp4s[CAM_NAMES[1]]), qpos_raw.shape[0])
    diag = np.asarray(traj["success"][:T]) if "success" in traj else None
    # obstacle contact per step is not in the h5; mark collision frame from the saved metric if present
    writer = None
    qmean = stats["qpos_mean"]; qstd = stats["qpos_std"]
    for t in range(T):
        qpos = (_decode_qpos(qpos_raw[t]) - qmean) / qstd
        qpos_t = torch.from_numpy(qpos).float().to(device).unsqueeze(0)
        cams_rgb, cams_bgr = [], {}
        for cam in CAM_NAMES:
            bgr = mp4s[cam][t]
            cams_bgr[cam] = bgr
            cams_rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
        image = np.transpose(np.stack(cams_rgb, 0), (0, 3, 1, 2))   # (ncam,3,H,W)
        image_t = torch.from_numpy(image).float().to(device).unsqueeze(0)
        prox_pos = None
        if extractor is not None:
            prox = _stack_prox(traj, t, sensor_order)               # (40,8,8)
            prox_pos = extractor(torch.from_numpy(prox).float().to(device).unsqueeze(0))
        with torch.inference_mode():
            _ = policy(qpos_t, image_t, proximity_positions=prox_pos)
        ov = _overlay(cap["attn"], n_prox, cams_bgr)
        panel = np.concatenate([ov[CAM_NAMES[0]], ov[CAM_NAMES[1]]], axis=1)  # exo | wrist
        cv2.putText(panel, f"{label}  ep{ep_idx}  t={t}/{T-1}  (FAIL)", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(panel, "exo", (8, IMG_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(panel, "wrist", (IMG_W + 8, IMG_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if writer is None:
            tmp_path = out_path.with_suffix(".tmp.mp4")
            writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     10.0, (panel.shape[1], panel.shape[0]))
        writer.write(panel)
    if writer is not None:
        writer.release()
        # OpenCV mp4v stutters in browsers/Slack; re-encode to H.264 (libx264).
        try:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_path),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                            "-movflags", "+faststart", str(out_path)], check=True)
            tmp_path.unlink(missing_ok=True)
        except Exception as e:
            print(f"[heatmap] ffmpeg re-encode failed ({e}); keeping mp4v")
            tmp_path.replace(out_path)
    handle.remove()
    return T


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True, type=Path)
    p.add_argument("--eval_dir", required=True, type=Path,
                   help="eval output dir containing house_*/trajectories*.h5 + episode MP4s")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--max_episodes", type=int, default=8, help="cap # failure episodes to render")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    policy, stats, extractor, n_prox, label = build_policy(args.ckpt_dir, args.device)

    h5s = sorted(args.eval_dir.glob("house_*/trajectories*.h5"))
    if not h5s:
        raise SystemExit(f"no trajectories h5 under {args.eval_dir}")
    n_done = 0
    for h5_path in h5s:
        house = h5_path.parent
        with h5py.File(h5_path, "r") as f:
            for key in sorted(f.keys(), key=lambda k: int(k.split("_")[1])):
                if n_done >= args.max_episodes:
                    break
                ep = int(key.split("_")[1])
                traj = f[key]
                succ = np.asarray(traj["success"][:]) if "success" in traj else np.array([False])
                failed = not bool(succ[-1])
                if not failed:
                    continue
                mp4s = {}
                ok = True
                for cam in CAM_NAMES:
                    cands = list(house.glob(f"episode_{ep:08d}_{cam}_batch_*.mp4"))
                    if not cands:
                        ok = False
                        break
                    mp4s[cam] = _read_mp4(cands[0])
                if not ok:
                    print(f"[heatmap] ep{ep}: missing MP4(s), skip")
                    continue
                out_path = args.out / f"ep{ep:03d}_attn.mp4"
                T = process_episode(policy, stats, extractor, n_prox, traj, mp4s,
                                    args.device, out_path, label, ep)
                print(f"[heatmap] {label} ep{ep}: wrote {out_path.name} (T={T})")
                n_done += 1
    print(f"[heatmap] DONE — {n_done} failure episodes -> {args.out}")


if __name__ == "__main__":
    main()
