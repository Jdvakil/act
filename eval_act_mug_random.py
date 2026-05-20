"""Rollout / evaluation script for the ACT policy trained on the
`mug_house_1_random_everything` dataset, against the *exact* datagen config
(`FrankaSkinPickAndPlacePilotMediumConfig`).

The datagen config draws a single episode (samples_per_house=1, house_inds=[1]).
That means each invocation of this script evaluates exactly one rollout. To
get aggregate statistics, launch this script 10× (or however many times you
want). Each Python process re-imports the config, and the import-time
``np.random.uniform(0.0, 1.0)`` calls inside
``FrankaSkinPickAndPlacePilotMediumConfig.task_sampler_config`` produce fresh
robot-object z-offset bounds per process — so the 10 runs cover the same
random-everything distribution the training set sampled from.

The eval config inherits from ``FrankaSkinPickAndPlacePilotMediumConfig`` and
overrides ONLY the policy attachment + bookkeeping fields (output dir, save
videos, etc). Nothing in the task sampler, seed, or randomization knobs is
changed.

Run from the repo root with the conda env that has molmospaces installed:

    cd /home/jaydv/code/prox_learning/submodules/act
    PYTHONPATH="$PWD:$PYTHONPATH" \\
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \\
    python eval_act_mug_random.py \\
        --ckpt_dir ckpts/act_house1_mug_random_v1 \\
        --output_dir /home/jaydv/code/prox_learning/eval_output/act_house1_mug_random_v1 \\
        --use_wandb
"""
from __future__ import annotations

# Force offscreen rendering before any mujoco / OpenGL import.
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.pop("DISPLAY", None)

import argparse
import pickle
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    import wandb  # type: ignore
except ImportError:
    wandb = None  # type: ignore

_WANDB_RUN = None
_ROLLOUT_INDEX = 0


@contextmanager
def _detr_argv(ckpt_dir: str, seed: int):
    """Shield DETR's main.py:get_args_parser from this script's CLI flags."""
    orig = sys.argv
    sys.argv = [
        orig[0] if orig else "eval_act_mug_random.py",
        "--ckpt_dir", ckpt_dir,
        "--policy_class", "ACT",
        "--task_name", "pla_house1_mug_random",
        "--seed", str(seed),
        "--num_epochs", "1",
    ]
    try:
        yield
    finally:
        sys.argv = orig


# ACT-side imports (we live inside submodules/act/)
from policy import ACTPolicy
from utils import set_seed

# molmospaces imports — eval target env / policy framework
from molmo_spaces.configs.policy_configs import BasePolicyConfig
from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
    FrankaSkinPickAndPlacePilotMediumConfig,
)
from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner
from molmo_spaces.policy.base_policy import InferencePolicy


# ----------------------------------------------------------------------
# Policy wrapper — identical to the one used in eval_act_house1.py.
# ----------------------------------------------------------------------
class ACTInferencePolicy(InferencePolicy):
    def __init__(self, exp_config, task=None) -> None:
        super().__init__(exp_config)
        self.task = task
        pc: ACTPolicyConfig = exp_config.policy_config
        self.pc = pc
        self.ckpt_path = str(Path(pc.ckpt_dir) / pc.ckpt_name)
        self.stats_path = str(Path(pc.ckpt_dir) / "dataset_stats.pkl")
        self._step: int = 0
        self._pending_chunks: list[tuple[int, np.ndarray]] = []
        self._policy = None
        self._stats = None

    def reset(self) -> None:
        global _ROLLOUT_INDEX
        if _WANDB_RUN is not None and self._step > 0:
            _WANDB_RUN.log(
                {
                    "rollout/episode_idx": _ROLLOUT_INDEX,
                    "rollout/episode_length": int(self._step),
                },
                step=_ROLLOUT_INDEX,
            )
            _ROLLOUT_INDEX += 1
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
        policy.cuda()
        policy.eval()
        self._policy = policy
        with open(self.stats_path, "rb") as f:
            self._stats = pickle.load(f)
        print(f"[act-eval] loaded {self.ckpt_path}")

    def obs_to_model_input(self, obs):
        if isinstance(obs, list | tuple):
            obs = obs[0]
        return obs

    def inference_model(self, obs):
        if self._policy is None:
            self.prepare_model()

        pc = self.pc
        stats = self._stats

        arm = np.asarray(obs["qpos"]["arm"][:7], dtype=np.float32)
        grip = np.asarray((obs["qpos"].get("gripper") or [0.0, 0.0])[:2], dtype=np.float32)
        qpos = np.concatenate([arm, grip], axis=0).astype(np.float32)
        qpos = (qpos - stats["qpos_mean"]) / stats["qpos_std"]
        qpos_t = torch.from_numpy(qpos).float().cuda().unsqueeze(0)

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
            a_hat = self._policy(qpos_t, image_t)
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


class ACTPolicyConfig(BasePolicyConfig):
    policy_cls: type = ACTInferencePolicy
    policy_type: str = "learned"

    ckpt_dir: str = ""
    ckpt_name: str = "policy_best.ckpt"
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


# ----------------------------------------------------------------------
# Eval config — inherits FrankaSkinPickAndPlacePilotMediumConfig verbatim
# (same task_sampler_config, same seed=2026, same randomize_lighting, same
# samples_per_house=1, same house_inds=[1]). Only swaps the policy in.
# ----------------------------------------------------------------------
class ACTMugRandomEvalConfig(FrankaSkinPickAndPlacePilotMediumConfig):
    """ACT eval against the same env used to collect mug_house_1_random_everything.

    Inherits every field of FrankaSkinPickAndPlacePilotMediumConfig — including
    the task sampler (samples_per_house=1, house_inds=[1]), seed=2026, and the
    randomize_lighting + random z-offset bound knobs — and only overrides what
    is needed to attach a learned policy and write videos / wandb output.

    Because samples_per_house=1, each invocation of the eval script runs ONE
    rollout. Aggregate over invocations to build statistics."""

    policy_config: ACTPolicyConfig = ACTPolicyConfig()
    use_wandb: bool = False
    filter_for_successful_trajectories: bool = False
    save_videos: bool = True
    use_passive_viewer: bool = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir",
                   default="/home/jaydv/code/prox_learning/submodules/act/ckpts/act_house1_mug_random_v1",
                   help="Directory containing policy_best.ckpt + dataset_stats.pkl")
    p.add_argument("--ckpt_name", default="policy_best.ckpt")
    p.add_argument("--output_dir",
                   default="/home/jaydv/code/prox_learning/eval_output/act_house1_mug_random_v1",
                   help="Where to write rollout MP4s + h5")
    p.add_argument("--task_horizon", type=int, default=300)
    p.add_argument("--chunk_size", type=int, default=100)
    p.add_argument("--kl_weight", type=int, default=10)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--dim_feedforward", type=int, default=3200)
    p.add_argument("--image_h", type=int, default=240)
    p.add_argument("--image_w", type=int, default=320)
    p.add_argument("--temp_agg_off", action="store_true")
    p.add_argument("--temp_agg_m", type=float, default=0.01)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="act-pla-house1-eval")
    p.add_argument("--wandb_run_name", type=str, default=None,
                   help="If omitted, auto-named eval_act_house1_mug_random_<unix_ts>")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_group", type=str, default="act_house1_mug_random_v1",
                   help="Group across the 10 manual invocations for aggregation in the UI")
    args, unknown = p.parse_known_args()
    if unknown:
        print(f"[act-eval] ignoring extra args: {unknown}")
    return args


def main() -> None:
    args = parse_args()

    # Build eval config (inherits config exactly; only policy attachment + IO).
    eval_cfg = ACTMugRandomEvalConfig()
    eval_cfg.task_horizon = args.task_horizon
    eval_cfg.output_dir = Path(args.output_dir).resolve()
    eval_cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Seed Python + torch with the same seed the env config uses so any
    # downstream code that reads numpy/torch global state is deterministic
    # for this run. The env config itself uses its own seed=2026 untouched.
    set_seed(int(eval_cfg.seed) if eval_cfg.seed is not None else 2026)

    pc = eval_cfg.policy_config
    pc.ckpt_dir = str(Path(args.ckpt_dir).resolve())
    pc.ckpt_name = args.ckpt_name
    pc.chunk_size = args.chunk_size
    pc.kl_weight = args.kl_weight
    pc.hidden_dim = args.hidden_dim
    pc.dim_feedforward = args.dim_feedforward
    pc.image_h = args.image_h
    pc.image_w = args.image_w
    pc.temp_agg_off = args.temp_agg_off
    pc.temp_agg_m = args.temp_agg_m

    eval_cfg.save_config()
    print(f"[act-eval] writing rollouts to {eval_cfg.output_dir}")
    print(f"[act-eval] task sampler samples_per_house={eval_cfg.task_sampler_config.samples_per_house}, "
          f"house_inds={eval_cfg.task_sampler_config.house_inds}, "
          f"seed={eval_cfg.seed}")

    # ----- wandb init -------------------------------------------------
    global _WANDB_RUN, _ROLLOUT_INDEX
    if args.use_wandb:
        if wandb is None:
            raise RuntimeError("--use_wandb passed but wandb is not installed.")
        run_name = args.wandb_run_name or f"eval_act_house1_mug_random_{int(time.time())}"
        _WANDB_RUN = wandb.init(
            project=args.wandb_project,
            name=run_name,
            entity=args.wandb_entity,
            group=args.wandb_group,
            config={
                "ckpt_dir": pc.ckpt_dir,
                "ckpt_name": pc.ckpt_name,
                "task_horizon": args.task_horizon,
                "chunk_size": pc.chunk_size,
                "kl_weight": pc.kl_weight,
                "hidden_dim": pc.hidden_dim,
                "dim_feedforward": pc.dim_feedforward,
                "image_h": pc.image_h,
                "image_w": pc.image_w,
                "temp_agg_off": pc.temp_agg_off,
                "temp_agg_m": pc.temp_agg_m,
                "env_config_class": "FrankaSkinPickAndPlacePilotMediumConfig",
                "env_seed": eval_cfg.seed,
                "samples_per_house": eval_cfg.task_sampler_config.samples_per_house,
                "house_inds": list(eval_cfg.task_sampler_config.house_inds),
            },
            tags=["act", "house1_mug_random", "eval",
                  "temp_agg_off" if pc.temp_agg_off else "temp_agg_on"],
        )
        _ROLLOUT_INDEX = 0
        print(f"[act-eval] wandb run: {_WANDB_RUN.url}  (group={args.wandb_group})")

    try:
        runner = ParallelRolloutRunner(eval_cfg)
        success, total = runner.run()
        print(f"[act-eval] success {success}/{total}")

        if _WANDB_RUN is not None:
            success_rate = (success / total) if total > 0 else 0.0
            _WANDB_RUN.log(
                {
                    "eval/success": int(success),
                    "eval/total": int(total),
                    "eval/success_rate": float(success_rate),
                }
            )
            _WANDB_RUN.summary["success"] = int(success)
            _WANDB_RUN.summary["total"] = int(total)
            _WANDB_RUN.summary["success_rate"] = float(success_rate)
            _log_rollout_videos_to_wandb(eval_cfg.output_dir)
    finally:
        if _WANDB_RUN is not None:
            _WANDB_RUN.finish()
            _WANDB_RUN = None


def _log_rollout_videos_to_wandb(output_dir: Path) -> None:
    if wandb is None or _WANDB_RUN is None:
        return
    house_dirs = sorted(p for p in output_dir.glob("house_*") if p.is_dir())
    n_logged = 0
    for house_dir in house_dirs:
        for mp4 in sorted(house_dir.glob("episode_*.mp4")):
            stem = mp4.stem
            parts = stem.split("_")
            try:
                ep_idx = int(parts[1])
            except (ValueError, IndexError):
                continue
            try:
                batch_pos = parts.index("batch")
                cam_name = "_".join(parts[2:batch_pos])
            except ValueError:
                cam_name = "_".join(parts[2:])
            key = f"videos/{house_dir.name}/ep{ep_idx:04d}/{cam_name}"
            try:
                _WANDB_RUN.log({key: wandb.Video(str(mp4), format="mp4")})
                n_logged += 1
            except Exception as e:
                print(f"[act-eval] could not log {mp4.name}: {e}")
    print(f"[act-eval] uploaded {n_logged} rollout videos to wandb")


if __name__ == "__main__":
    main()
