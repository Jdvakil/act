"""Standalone rollout / evaluation script for an ACT checkpoint trained on
the PLA house_10 cup pick-and-place task (1:1 match with the datagen config
`FrankaSkinPickAndPlaceHouse10CupConfig`).

This script:
  1) Loads the ACT checkpoint and dataset_stats.pkl produced by
     `imitate_episodes.py`.
  2) Instantiates the EXACT same molmospaces environment used for data
     generation (`FrankaSkinPickAndPlaceHouse10CupConfig`) — house_10,
     cup pickup, no visibility check — and attaches the learned policy.
  3) Rolls out N episodes with temporal-ensembling action chunking (Zhao et al.
     2023) and writes per-episode MP4 videos + an h5 trajectory bundle into
     `--output_dir`.

Run from the repo root with the conda env that has molmospaces installed:

    cd /home/jaydv/code/prox_learning/submodules/act
    PYTHONPATH="$PWD:$PYTHONPATH" \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    python eval_act_house10_cup.py \
        --ckpt_dir /path/to/training/ckpt_dir \
        --num_rollouts 10 \
        --output_dir /home/jaydv/code/prox_learning/eval_output/act_house10_cup
"""
from __future__ import annotations

# Force offscreen rendering before any mujoco / OpenGL import — required when
# running over SSH with no display attached. Set BEFORE molmospaces / mujoco
# imports below.
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
# Make sure nothing tries to open a display.
os.environ.pop("DISPLAY", None)

import argparse
import pickle
import sys
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    import wandb  # type: ignore
except ImportError:  # wandb is optional — eval still works without it.
    wandb = None  # type: ignore

# Module-level handle so ACTInferencePolicy.reset() can log live progress
# without us having to wire wandb through pydantic config or the runner.
_WANDB_RUN = None
_ROLLOUT_INDEX = 0


@contextmanager
def _detr_argv(ckpt_dir: str, seed: int):
    """Temporarily replace sys.argv with the minimal set DETR's parse_args needs.

    detr/main.py:get_args_parser() declares --ckpt_dir / --policy_class /
    --task_name / --seed / --num_epochs as required and parses sys.argv when
    ACTPolicy builds the model. The eval CLI carries unrelated flags (e.g.
    --num_rollouts, --output_dir), so we swap sys.argv for the duration of
    the model build to keep DETR happy without leaking our flags into it."""
    orig = sys.argv
    sys.argv = [
        orig[0] if orig else "eval_act_house1.py",
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

# ACT-side imports (we live inside submodules/act/)
from policy import ACTPolicy
from utils import set_seed

# molmospaces imports — eval target env / policy framework
from molmo_spaces.configs.policy_configs import BasePolicyConfig
from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
    FrankaSkinPickAndPlaceHouse10CupConfig,
)
from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner
from molmo_spaces.policy.base_policy import InferencePolicy
from molmo_spaces.tasks.pick_and_place_task_sampler import PickAndPlaceTaskSampler


# ----------------------------------------------------------------------
# Task sampler: re-populate `candidate_objects` on every attempt.
#
# The upstream PickTaskSampler treats `_remove_candidate_object` as a
# permanent blacklist for the lifetime of the loaded scene. This is fine
# for datagen across hundreds of houses with many pickup candidates, but
# in single-house single-task eval (`OneHouseMugConfig`: house_1 has only
# 2 mug candidates) a single bad texture/dynamics roll that breaks robot
# placement visibility can wipe both candidates and leave the sampler with
# `candidate_objects=[]` forever — every subsequent attempt then trips the
# assertion in `_sample_task` and counts as `task_sampling_failed`, yielding
# 1-2 evaluated rollouts out of 1000 attempts.
#
# Eval semantics want each attempt to be independent: re-randomize the
# scene and re-evaluate all candidates from scratch.
# ----------------------------------------------------------------------
class PickAndPlaceEvalTaskSampler(PickAndPlaceTaskSampler):
    """Variant that re-populates `candidate_objects` before each sample."""

    def _sample_task(self, env):
        candidates = self._get_scene_objects(env)
        candidates = self.balance_sample_names(candidates)
        np.random.shuffle(candidates)
        self.candidate_objects = candidates
        # Also reset per-object grasp-failure counters so a transient bad
        # randomization doesn't permanently disqualify an object.
        self._grasp_failure_counts = {}
        return super()._sample_task(env)


# ----------------------------------------------------------------------
# Policy wrapper that plugs a trained ACT checkpoint into the molmospaces
# rollout pipeline.
# ----------------------------------------------------------------------
class ACTInferencePolicy(InferencePolicy):
    """Wraps a trained ACT checkpoint as a molmospaces InferencePolicy."""

    def __init__(self, exp_config, task=None) -> None:
        super().__init__(exp_config)
        self.task = task
        pc: ACTPolicyConfig = exp_config.policy_config
        self.pc = pc
        # Resolve checkpoint paths once.
        self.ckpt_path = str(Path(pc.ckpt_dir) / pc.ckpt_name)
        self.stats_path = str(Path(pc.ckpt_dir) / "dataset_stats.pkl")
        # Per-episode chunk-cache state for temporal ensembling.
        self._step: int = 0
        self._pending_chunks: list[tuple[int, np.ndarray]] = []
        # Lazy model load.
        self._policy = None
        self._stats = None

    # ----- abstract methods from InferencePolicy --------------------
    def reset(self) -> None:
        # Live wandb signal: log the *previous* episode's length (the policy
        # doesn't see success/failure here, but step count is a useful proxy
        # for whether a rollout terminated early vs ran to the horizon).
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
        # Build model and load weights. DETR's main.py parses sys.argv inside
        # build_ACT_model_and_optimizer; shield it from this script's flags.
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

        # --- qpos (1, 9) -------------------------------------------------
        arm = np.asarray(obs["qpos"]["arm"][:7], dtype=np.float32)
        grip = np.asarray((obs["qpos"].get("gripper") or [0.0, 0.0])[:2], dtype=np.float32)
        qpos = np.concatenate([arm, grip], axis=0).astype(np.float32)  # (9,)
        qpos = (qpos - stats["qpos_mean"]) / stats["qpos_std"]
        qpos_t = torch.from_numpy(qpos).float().cuda().unsqueeze(0)  # (1, 9)

        # --- images (1, num_cam, 3, H, W) in [0, 1] ----------------------
        cams = []
        for cam in pc.camera_names:
            img = obs[cam]
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            if img.shape[:2] != (pc.image_h, pc.image_w):
                img = cv2.resize(img, (pc.image_w, pc.image_h), interpolation=cv2.INTER_AREA)
            cams.append(img.astype(np.float32) / 255.0)
        image = np.stack(cams, axis=0)                       # (num_cam, H, W, 3)
        image = np.transpose(image, (0, 3, 1, 2))             # (num_cam, 3, H, W)
        image_t = torch.from_numpy(image).float().cuda().unsqueeze(0)  # (1, num_cam, 3, H, W)

        # --- ACT forward (no actions = inference mode) ------------------
        with torch.no_grad():
            a_hat = self._policy(qpos_t, image_t)  # (1, chunk_size, action_dim) normalized
        new_chunk = a_hat.squeeze(0).cpu().numpy()           # (H, A)
        # Un-normalize once.
        new_chunk = new_chunk * stats["action_std"] + stats["action_mean"]

        if pc.temp_agg_off:
            # No ensembling: replay the whole chunk before re-querying.
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
        # model_output: (action_dim,) un-normalized → arm(7) + gripper_cmd(1)
        arm = np.asarray(model_output[:7], dtype=np.float32)
        gripper_raw = float(model_output[7]) if len(model_output) >= 8 else 0.0
        # Datagen records gripper as exactly {0.0, 255.0}. Snap to binary.
        gripper = 0.0 if gripper_raw < 127.5 else 255.0
        return {"arm": arm, "gripper": np.asarray([gripper], dtype=np.float32)}

    def get_action(self, obs):
        action = super().get_action(obs)
        self._step += 1
        return action


class ACTPolicyConfig(BasePolicyConfig):
    """Pydantic policy config consumed by molmospaces' ParallelRolloutRunner.

    The runner calls `policy_cls(exp_config, task)` to instantiate the policy
    inside each worker (see data_generation/pipeline.py:setup_policy)."""

    policy_cls: type = ACTInferencePolicy
    policy_type: str = "learned"

    ckpt_dir: str = ""
    ckpt_name: str = "policy_best.ckpt"
    image_h: int = 240
    image_w: int = 320
    camera_names: tuple[str, ...] = ("exo_camera_1", "wrist_camera")
    # Action chunking / temporal ensembling.
    chunk_size: int = 100
    temp_agg_m: float = 0.01
    temp_agg_off: bool = False
    # ACT model hyperparams — must match training. Defaults to the
    # `--kl_weight 10 --hidden_dim 512 --dim_feedforward 3200` set from the
    # ACT README and the values used in imitate_episodes.py.
    kl_weight: int = 10
    hidden_dim: int = 512
    dim_feedforward: int = 3200
    enc_layers: int = 4
    dec_layers: int = 7
    nheads: int = 8
    # State / action dims.
    state_dim: int = 9   # arm(7) + 2 finger joints
    action_dim: int = 8  # arm(7) + gripper_cmd(1)
    backbone: str = "resnet18"
    lr: float = 1e-5
    lr_backbone: float = 1e-5
    seed: int = 0


# ----------------------------------------------------------------------
# Eval config — house_10 cup pickup with our policy instead of the
# scripted teleop planner. 1:1 match with FrankaSkinPickAndPlaceHouse10CupConfig.
# ----------------------------------------------------------------------
class ACTHouse10CupEvalConfig(FrankaSkinPickAndPlaceHouse10CupConfig):
    """Eval-time variant of the house_10 cup datagen config.

    Inherits the env, task sampler, robot, cameras, and task horizon from the
    original datagen config; swaps the scripted policy for the learned ACT
    checkpoint and turns off success-only filtering so failed rollouts are
    still saved for visual inspection."""

    policy_config: ACTPolicyConfig = ACTPolicyConfig()
    use_wandb: bool = False
    filter_for_successful_trajectories: bool = False
    save_videos: bool = True
    num_workers: int = 1  # ACT eval is GPU-bound; single worker keeps it simple.
    # Force offscreen rendering — eval runs headless over SSH.
    use_passive_viewer: bool = False

    def model_post_init(self, __context):
        super().model_post_init(__context)
        # No injected action noise during evaluation.
        if hasattr(self.robot_config, "action_noise_config"):
            self.robot_config.action_noise_config.enabled = False


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True, help="Directory containing policy_best.ckpt + dataset_stats.pkl")
    p.add_argument("--ckpt_name", default="policy_best.ckpt")
    p.add_argument("--output_dir", required=True, help="Where to write rollout MP4s + h5")
    p.add_argument("--num_rollouts", type=int, default=10)
    p.add_argument("--task_horizon", type=int, default=300)
    p.add_argument("--chunk_size", type=int, default=100)
    p.add_argument("--kl_weight", type=int, default=10)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--dim_feedforward", type=int, default=3200)
    p.add_argument("--image_h", type=int, default=240)
    p.add_argument("--image_w", type=int, default=320)
    p.add_argument("--temp_agg_off", action="store_true", help="Disable temporal ensembling.")
    p.add_argument("--temp_agg_m", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    # wandb logging.
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="act-pla-house10-cup-eval")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    # Tolerate DETR-only flags (--policy_class, --task_name, --num_epochs) that
    # a user might paste from the training command. The _detr_argv shim
    # provides the values DETR's parser actually needs.
    args, unknown = p.parse_known_args()
    if unknown:
        print(f"[act-eval] ignoring extra args: {unknown}")
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Build the eval config and override fields from CLI.
    eval_cfg = ACTHouse10CupEvalConfig()
    eval_cfg.task_horizon = args.task_horizon
    eval_cfg.task_sampler_config.samples_per_house = args.num_rollouts
    # Limit total rollouts up front (the sampler will run until quotas hit).
    eval_cfg.output_dir = Path(args.output_dir).resolve()
    eval_cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # The task sampler runs the scripted teleop planner first to verify each
    # randomized scene is feasible (IK preplace, grasp pose, etc) and rejects
    # the rest as HouseInvalidForTask. Default multiplier=12 (so 12*num_rollouts
    # attempts) often runs out before we collect num_rollouts feasible scenes.
    # Bump it generously — wasted attempts are cheap compared to GPU rollout
    # time, and we want every requested rollout to actually run.
    eval_cfg.task_sampler_config.max_total_attempts_multiplier = 100
    eval_cfg.task_sampler_config.max_allowed_sequential_task_sampler_failures = 2000

    # Replace the upstream sampler with the eval variant that re-populates
    # `candidate_objects` per attempt (see class docstring above for why).
    eval_cfg.task_sampler_config.task_sampler_class = PickAndPlaceEvalTaskSampler

    # Push CLI overrides into the policy config.
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

    # ----- wandb init -------------------------------------------------
    global _WANDB_RUN, _ROLLOUT_INDEX
    if args.use_wandb:
        if wandb is None:
            raise RuntimeError("--use_wandb passed but wandb is not installed.")
        _WANDB_RUN = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config={
                "ckpt_dir": pc.ckpt_dir,
                "ckpt_name": pc.ckpt_name,
                "num_rollouts": args.num_rollouts,
                "task_horizon": args.task_horizon,
                "chunk_size": pc.chunk_size,
                "kl_weight": pc.kl_weight,
                "hidden_dim": pc.hidden_dim,
                "dim_feedforward": pc.dim_feedforward,
                "image_h": pc.image_h,
                "image_w": pc.image_w,
                "temp_agg_off": pc.temp_agg_off,
                "temp_agg_m": pc.temp_agg_m,
                "seed": args.seed,
                "max_total_attempts_multiplier": eval_cfg.task_sampler_config.max_total_attempts_multiplier,
            },
            tags=["act", "house10_cup", "eval", "temp_agg_off" if pc.temp_agg_off else "temp_agg_on"],
        )
        _ROLLOUT_INDEX = 0
        print(f"[act-eval] wandb run: {_WANDB_RUN.url}")

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
    """After the runner finishes, find per-episode MP4s and log them.

    The runner writes `<output_dir>/house_<id>/episode_<NNNNNNNN>_<cam>_batch_*.mp4`
    once the full batch completes. We log each camera as a separate wandb.Video
    and key them by episode index so they show up grouped in the run UI."""
    if wandb is None or _WANDB_RUN is None:
        return
    house_dirs = sorted(p for p in output_dir.glob("house_*") if p.is_dir())
    n_logged = 0
    for house_dir in house_dirs:
        for mp4 in sorted(house_dir.glob("episode_*.mp4")):
            # Filename: episode_00000000_<camera>_batch_1_of_1.mp4
            stem = mp4.stem
            parts = stem.split("_")
            try:
                ep_idx = int(parts[1])
            except (ValueError, IndexError):
                continue
            # camera name is everything between ep_idx and 'batch'
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
