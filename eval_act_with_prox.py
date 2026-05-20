"""Eval ACT + ProximityResidualHead together. Same molmospaces rollout pipeline
as `eval_act_house1.py`, but at each chunk re-query the proximity head and add
its residual to the ACT chunk before passing into the temporal-ensembling
buffer.

Run:
    cd /home/jaydv/code/prox_learning/submodules/act

    # Baseline (no head):
    PYTHONPATH="$PWD:$PWD/detr:/home/jaydv/code/prox_learning:$PYTHONPATH" \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    python eval_act_with_prox.py \
        --ckpt_dir ckpts/act_house1_mug_v3 \
        --output_dir /home/jaydv/code/prox_learning/eval_output/act_v3_baseline \
        --num_rollouts 10

    # With proximity residual head:
    PYTHONPATH="$PWD:$PWD/detr:/home/jaydv/code/prox_learning:$PYTHONPATH" \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    python eval_act_with_prox.py \
        --ckpt_dir ckpts/act_house1_mug_v3 \
        --prox_head_ckpt /home/jaydv/code/prox_learning/runs/prox_residual_v1/head_best.pt \
        --output_dir /home/jaydv/code/prox_learning/eval_output/act_v3_plus_prox \
        --num_rollouts 10
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.pop("DISPLAY", None)

import argparse
import pickle
import sys
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import torch

# Make pla.prox_residual_head importable.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policy import ACTPolicy
from utils import set_seed

from molmo_spaces.configs.policy_configs import BasePolicyConfig
from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
    FrankaSkinPickAndPlaceOneHouseMugConfig,
    FrankaSkinPickAndPlacePilotSmokeConfig,
)
from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner
from molmo_spaces.policy.base_policy import InferencePolicy

from pla.prox_residual_head import ProximityResidualHead
from pla.prox_residual_dataset import SENSOR_NAMES


@contextmanager
def _detr_argv(ckpt_dir: str, seed: int):
    orig = sys.argv
    sys.argv = [
        orig[0] if orig else "eval_act_with_prox.py",
        "--ckpt_dir", ckpt_dir,
        "--policy_class", "ACT",
        "--task_name", "pla_house1_mug",
        "--seed", str(seed),
        "--num_epochs", "1",
    ]
    try:
        yield
    finally:
        sys.argv = orig


class ACTWithProxInferencePolicy(InferencePolicy):
    def __init__(self, exp_config, task=None) -> None:
        super().__init__(exp_config)
        self.task = task
        pc: ACTProxPolicyConfig = exp_config.policy_config
        self.pc = pc
        self.ckpt_path = str(Path(pc.ckpt_dir) / pc.ckpt_name)
        self.stats_path = str(Path(pc.ckpt_dir) / "dataset_stats.pkl")
        self._step: int = 0
        self._pending_chunks: list[tuple[int, np.ndarray]] = []
        self._policy = None
        self._head = None
        self._stats = None

    def reset(self) -> None:
        self._step = 0
        self._pending_chunks.clear()

    def prepare_model(self, model_name: str | None = None) -> None:
        pc = self.pc
        policy_config = {
            "lr": pc.lr,
            "num_queries": pc.chunk_size,
            "kl_weight": pc.kl_weight,
            "hidden_dim": pc.hidden_dim,
            "dim_feedforward": pc.dim_feedforward,
            "lr_backbone": pc.lr_backbone,
            "backbone": pc.backbone,
            "enc_layers": pc.enc_layers,
            "dec_layers": pc.dec_layers,
            "nheads": pc.nheads,
            "camera_names": list(pc.camera_names),
            "state_dim": pc.state_dim,
            "action_dim": pc.action_dim,
        }
        with _detr_argv(self.pc.ckpt_dir, self.pc.seed):
            policy = ACTPolicy(policy_config)
        sd = torch.load(self.ckpt_path, map_location="cuda")
        policy.load_state_dict(sd)
        policy.cuda().eval()
        self._policy = policy

        with open(self.stats_path, "rb") as f:
            self._stats = pickle.load(f)

        if pc.prox_head_ckpt:
            head = ProximityResidualHead(
                chunk_size=pc.chunk_size,
                action_dim=pc.action_dim,
                qpos_dim=pc.state_dim,
                mask_links=tuple(pc.mask_links),
            )
            head.load_state_dict(torch.load(pc.prox_head_ckpt, map_location="cuda"))
            head.cuda().eval()
            self._head = head
            print(f"[act-prox] loaded residual head {pc.prox_head_ckpt}")
        else:
            self._head = None
            print(f"[act-prox] running BASELINE (no residual head)")
        print(f"[act-prox] loaded ACT {self.ckpt_path}")

    def obs_to_model_input(self, obs):
        if isinstance(obs, list | tuple):
            obs = obs[0]
        return obs

    def _build_proximity(self, obs) -> torch.Tensor:
        stack = np.zeros((29, 8, 8), dtype=np.float32)
        for i, sname in enumerate(SENSOR_NAMES):
            arr = np.asarray(obs[sname], dtype=np.float32)
            if arr.ndim == 3:  # (n_substeps, 8, 8)
                arr = arr.mean(axis=0)
            stack[i] = arr
        stack = np.clip(stack / float(self.pc.depth_max_m), 0.0, 1.0)
        return torch.from_numpy(stack).unsqueeze(0).cuda()  # (1, 29, 8, 8)

    def inference_model(self, obs):
        if self._policy is None:
            self.prepare_model()
        pc = self.pc
        stats = self._stats

        arm = np.asarray(obs["qpos"]["arm"][:7], dtype=np.float32)
        grip = np.asarray((obs["qpos"].get("gripper") or [0.0, 0.0])[:2], dtype=np.float32)
        qpos_np = np.concatenate([arm, grip], axis=0).astype(np.float32)
        qpos_norm = (qpos_np - stats["qpos_mean"]) / stats["qpos_std"]
        qpos_t = torch.from_numpy(qpos_norm).float().cuda().unsqueeze(0)

        cams = []
        for cam in pc.camera_names:
            img = obs[cam]
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            if img.shape[:2] != (pc.image_h, pc.image_w):
                img = cv2.resize(img, (pc.image_w, pc.image_h), interpolation=cv2.INTER_AREA)
            cams.append(img.astype(np.float32) / 255.0)
        image = np.stack(cams, axis=0)
        image = np.transpose(image, (0, 3, 1, 2))
        image_t = torch.from_numpy(image).float().cuda().unsqueeze(0)

        with torch.no_grad():
            a_hat = self._policy(qpos_t, image_t)            # (1, chunk, A) normalized
            if self._head is not None:
                prox_t = self._build_proximity(obs)           # (1, 29, 8, 8)
                delta = self._head(prox_t, qpos_t)            # (1, chunk, A) normalized
                a_hat = a_hat + delta

        new_chunk = a_hat.squeeze(0).cpu().numpy()
        new_chunk = new_chunk * stats["action_std"] + stats["action_mean"]

        if pc.temp_agg_off:
            self._pending_chunks = [(self._step, new_chunk)]
            return new_chunk[0]
        H = new_chunk.shape[0]
        self._pending_chunks.append((self._step, new_chunk))
        self._pending_chunks = [
            (s, c) for (s, c) in self._pending_chunks if self._step - s < H
        ]
        preds, weights = [], []
        for (start, chunk) in self._pending_chunks:
            k = self._step - start
            if 0 <= k < H:
                preds.append(chunk[k])
                weights.append(np.exp(-pc.temp_agg_m * k))
        preds_a = np.stack(preds, axis=0)
        w = np.asarray(weights, dtype=np.float64)
        w /= w.sum()
        return (preds_a * w[:, None]).sum(axis=0).astype(np.float32)

    def model_output_to_action(self, model_output):
        arm = np.asarray(model_output[:7], dtype=np.float32)
        gripper_raw = float(model_output[7]) if len(model_output) >= 8 else 0.0
        gripper = 0.0 if gripper_raw < 127.5 else 255.0
        return {"arm": arm, "gripper": np.asarray([gripper], dtype=np.float32)}

    def get_action(self, obs):
        action = super().get_action(obs)
        self._step += 1
        return action


class ACTProxPolicyConfig(BasePolicyConfig):
    policy_cls: type = ACTWithProxInferencePolicy
    policy_type: str = "learned"

    ckpt_dir: str = ""
    ckpt_name: str = "policy_best.ckpt"
    prox_head_ckpt: str = ""
    mask_links: tuple[str, ...] = ("link2",)
    depth_max_m: float = 4.0
    image_h: int = 240
    image_w: int = 320
    camera_names: tuple[str, ...] = ("exo_camera_1", "wrist_camera")
    chunk_size: int = 100
    temp_agg_m: float = 0.01
    temp_agg_off: bool = False
    kl_weight: int = 10
    hidden_dim: int = 512
    dim_feedforward: int = 3200
    enc_layers: int = 4
    dec_layers: int = 7
    nheads: int = 8
    state_dim: int = 9
    action_dim: int = 8
    backbone: str = "resnet18"
    lr: float = 1e-5
    lr_backbone: float = 1e-5
    seed: int = 0


class ACTProxHouse1MugEvalConfig(FrankaSkinPickAndPlaceOneHouseMugConfig):
    policy_config: ACTProxPolicyConfig = ACTProxPolicyConfig()
    use_wandb: bool = False
    filter_for_successful_trajectories: bool = False
    save_videos: bool = True
    num_workers: int = 1
    use_passive_viewer: bool = False


class ACTProxSmokeEvalConfig(FrankaSkinPickAndPlacePilotSmokeConfig):
    """Eval against the smoke distribution (10 houses, multi-object pick-and-place)
    that the ACT + proximity-residual head was trained on."""

    policy_config: ACTProxPolicyConfig = ACTProxPolicyConfig()
    use_wandb: bool = False
    filter_for_successful_trajectories: bool = False
    save_videos: bool = True
    num_workers: int = 1
    use_passive_viewer: bool = False


_BASE_CONFIGS = {
    "house1_mug": ACTProxHouse1MugEvalConfig,
    "smoke": ACTProxSmokeEvalConfig,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--ckpt_name", default="policy_best.ckpt")
    p.add_argument("--prox_head_ckpt", default="",
                   help="Path to head_best.pt. Empty = baseline (no head).")
    p.add_argument("--mask_links", nargs="*", default=["link2"])
    p.add_argument("--depth_max_m", type=float, default=4.0)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_rollouts", type=int, default=10)
    p.add_argument("--task_horizon", type=int, default=300)
    p.add_argument("--chunk_size", type=int, default=100)
    p.add_argument("--kl_weight", type=int, default=10)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--dim_feedforward", type=int, default=3200)
    p.add_argument("--image_h", type=int, default=240)
    p.add_argument("--image_w", type=int, default=320)
    p.add_argument("--temp_agg_off", action="store_true")
    p.add_argument("--temp_agg_m", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--base_config", default="house1_mug", choices=list(_BASE_CONFIGS),
                   help="Which env config to roll out under.")
    p.add_argument("--house_inds", nargs="*", type=int, default=None,
                   help="Override task_sampler_config.house_inds (multi-house configs).")
    args, unknown = p.parse_known_args()
    if unknown:
        print(f"[act-prox] ignoring extra args: {unknown}")
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    cfg_cls = _BASE_CONFIGS[args.base_config]
    cfg = cfg_cls()
    cfg.task_horizon = args.task_horizon
    cfg.task_sampler_config.samples_per_house = args.num_rollouts
    if args.house_inds is not None:
        cfg.task_sampler_config.house_inds = list(args.house_inds)
    cfg.output_dir = Path(args.output_dir).resolve()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.seed = args.seed

    pc = cfg.policy_config
    pc.ckpt_dir = str(Path(args.ckpt_dir).resolve())
    pc.ckpt_name = args.ckpt_name
    pc.prox_head_ckpt = (
        str(Path(args.prox_head_ckpt).resolve()) if args.prox_head_ckpt else ""
    )
    pc.mask_links = tuple(args.mask_links)
    pc.depth_max_m = float(args.depth_max_m)
    pc.chunk_size = args.chunk_size
    pc.kl_weight = args.kl_weight
    pc.hidden_dim = args.hidden_dim
    pc.dim_feedforward = args.dim_feedforward
    pc.image_h = args.image_h
    pc.image_w = args.image_w
    pc.temp_agg_off = args.temp_agg_off
    pc.temp_agg_m = args.temp_agg_m

    cfg.save_config()
    runner = ParallelRolloutRunner(cfg)
    success, total = runner.run()
    rate = (success / total) if total else 0.0
    label = "PROX_HEAD" if pc.prox_head_ckpt else "BASELINE"
    print(f"[act-prox] [{label}] success {success}/{total} ({rate:.1%})")


if __name__ == "__main__":
    main()
