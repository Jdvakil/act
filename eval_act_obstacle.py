"""In-env rollout / evaluation for the VANILLA ACT baseline trained on the one-env
obstacle pick (`obstacle_baseline` task in constants.py).

The whole point of this script is to evaluate ACT *in the exact environment the data
was collected in*. It does that by inheriting the datagen config
`FrankaSkinHybridObstacleConfig` verbatim — same custom fumehood scene, same FR3
hybrid-skin robot, same exo/wrist cameras, same `ObstacleFumehoodPickSampler` (red
cup, hazard bar present ~75% of episodes) — and swapping ONLY the policy: the scripted
`ObstacleAwarePickPlannerPolicy` is replaced by `ACTInferencePolicy`, which loads the
trained ACT checkpoint and drives the arm from (rgb + qpos). No proximity / skin input
is used — this is the no-proximity baseline.

The molmospaces datagen pipeline (`ParallelRolloutRunner`) is reused unchanged, so task
sampling, the rollout loop, success judging (`PickTask.judge_success`) and video saving
are all identical to collection. With `filter_for_successful_trajectories=False`, every
rollout (success or failure) counts toward `samples_per_house`, so one invocation runs
exactly `--num_rollouts` episodes and `runner.run()` returns (successes, total).

Rendering has two independent layers and this script exposes both:
  * Camera observations + saved rollout MP4s ALWAYS render offscreen via EGL on the
    headless GPU (molmospaces forces MUJOCO_GL=egl at import; nothing to configure).
  * A live MuJoCo viewer window (mujoco.viewer.launch_passive) can additionally be
    opened with `--live`. It uses its own GLFW window — independent of MUJOCO_GL — so
    it needs an attached display ($DISPLAY / $WAYLAND_DISPLAY) and a single-process
    run. Omit `--live` (the default) on a headless box and you get clean offscreen-only
    behavior; pass `--live` at your desktop to watch the policy roll out in real time.

Run from the act submodule dir with the conda env that has molmospaces installed:

    # headless (default) — offscreen render + MP4s only
    cd /home/jaydv/code/prox_learning/submodules/act
    PYTHONPATH="$PWD:$PYTHONPATH" \\
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \\
    python eval_act_obstacle.py \\
        --ckpt_dir ckpts/act_obstacle_baseline_v1 \\
        --output_dir /home/jaydv/code/prox_learning/eval_output/act_obstacle_baseline_v1 \\
        --num_rollouts 25

    # desktop — same, plus a live viewer window (add --live; do NOT unset DISPLAY)
    python eval_act_obstacle.py --ckpt_dir ckpts/... --num_rollouts 5 --live
"""
from __future__ import annotations

# Rendering backend must be chosen BEFORE any mujoco / OpenGL import.
#
# Offscreen camera rendering + video saving always run on EGL (molmospaces forces
# MUJOCO_GL=egl at import regardless), and the optional live passive viewer opens its
# OWN GLFW window that does NOT consult MUJOCO_GL. So EGL stays on in both modes; the
# only difference is whether we keep $DISPLAY (GLFW needs it to open the live window)
# or drop it for a clean headless run. We peek at argv here because this decision must
# be made before molmospaces is imported.
import os
import sys

_LIVE_RENDER = any(a in ("--live", "--render", "--viewer") for a in sys.argv[1:])

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
if not _LIVE_RENDER:
    # Headless: drop any inherited X11 display so nothing tries to open a window.
    os.environ.pop("DISPLAY", None)

import argparse
import pickle
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
        orig[0] if orig else "eval_act_obstacle.py",
        "--ckpt_dir", ckpt_dir,
        "--policy_class", "ACT",
        "--task_name", "obstacle_baseline",
        "--seed", str(seed),
        "--num_epochs", "1",
    ]
    try:
        yield
    finally:
        sys.argv = orig


# ACT-side imports (we live inside submodules/act/).
from policy import ACTPolicy
from utils import set_seed

# molmospaces imports — eval target env / policy framework.
from molmo_spaces.configs.policy_configs import BasePolicyConfig
from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
    FrankaSkinHybridObstacleConfig,
)
from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.policy.base_policy import InferencePolicy


# ----------------------------------------------------------------------
# Policy wrapper — identical pattern to eval_act_mug_random.py. Pulls (rgb + qpos)
# out of the live observation, runs ACT, optionally temporally aggregates, and emits
# the {"arm", "gripper"} action dict the FR3 robot consumes.
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

        # Temporal aggregation (ACT default): exponentially weight all in-flight
        # chunk predictions for the current step.
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
        # Training gripper command is the FR3 hand actuator at {0, 255}; snap the
        # continuous prediction back to the nearest extreme.
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
# Eval config — inherits FrankaSkinHybridObstacleConfig verbatim (same scene, robot,
# cameras, and ObstacleFumehoodPickSampler) and only swaps the policy + IO. The task
# sampler's samples_per_house / house_inds / num_workers are set per-run in main().
# ----------------------------------------------------------------------
class ACTObstacleEvalConfig(FrankaSkinHybridObstacleConfig):
    """ACT eval against the exact env used to collect hybrid_obstacle_v1."""

    policy_config: ACTPolicyConfig = ACTPolicyConfig()
    num_workers: int = 1
    use_wandb: bool = False
    filter_for_successful_trajectories: bool = False
    save_videos: bool = True
    use_passive_viewer: bool = False
    output_dir: Path = ASSETS_DIR / "datagen" / "act_obstacle_eval"

    @property
    def tag(self) -> str:
        return "act_obstacle_eval"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt_dir",
        default="/home/jaydv/code/prox_learning/submodules/act/ckpts/act_obstacle_baseline_v1",
        help="Directory containing policy_best.ckpt + dataset_stats.pkl",
    )
    p.add_argument("--ckpt_name", default="policy_best.ckpt")
    p.add_argument(
        "--output_dir",
        default="/home/jaydv/code/prox_learning/eval_output/act_obstacle_baseline_v1",
        help="Where to write rollout MP4s + h5",
    )
    p.add_argument("--num_rollouts", type=int, default=25,
                   help="Episodes to roll out in this process (samples_per_house).")
    p.add_argument("--house_ind", type=int, default=1,
                   help="ProcTHOR house index; 1 (==1 mod 24) is the red cup the data used.")
    p.add_argument("--task_horizon", type=int, default=200,
                   help="Max policy steps per episode (source episodes were ~84, max 167).")
    p.add_argument("--chunk_size", type=int, default=100)
    p.add_argument("--kl_weight", type=int, default=10)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--dim_feedforward", type=int, default=3200)
    p.add_argument("--image_h", type=int, default=240)
    p.add_argument("--image_w", type=int, default=320)
    p.add_argument("--temp_agg_off", action="store_true")
    p.add_argument("--temp_agg_m", type=float, default=0.01)
    p.add_argument(
        "--live", "--render", "--viewer",
        dest="live",
        action="store_true",
        help="Open a live MuJoCo viewer window during eval (desktop only; needs "
             "$DISPLAY/$WAYLAND_DISPLAY). Default off = clean headless offscreen "
             "rendering. Forces single-process rollout when on.",
    )
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="act-obstacle-baseline-eval")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_group", type=str, default="act_obstacle_baseline_v1")
    args, unknown = p.parse_known_args()
    if unknown:
        print(f"[act-eval] ignoring extra args: {unknown}")
    return args


def main() -> None:
    args = parse_args()

    eval_cfg = ACTObstacleEvalConfig()
    eval_cfg.task_horizon = args.task_horizon
    eval_cfg.output_dir = Path(args.output_dir).resolve()
    eval_cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Drive N rollouts of the single red-cup task from one process.
    eval_cfg.task_sampler_config.samples_per_house = args.num_rollouts
    eval_cfg.task_sampler_config.house_inds = [args.house_ind]

    # Live viewer (desktop): keep $DISPLAY (already kept by the argv peek at import) and
    # ask the pipeline to launch the passive GLFW window. The window must live in the
    # main process, so force single-worker. Falls back to headless if no display.
    live = args.live
    if live and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print(
            "[act-eval] --live requested but no $DISPLAY/$WAYLAND_DISPLAY found; "
            "falling back to headless offscreen rendering. Run on your desktop "
            "session to see the live viewer."
        )
        live = False
    eval_cfg.use_passive_viewer = live
    if live:
        eval_cfg.num_workers = 1
        print(
            "[act-eval] LIVE render ON — a MuJoCo viewer window will open "
            "(single-process). Playback runs as fast as the sim/policy allow; "
            "MP4s are still written to --output_dir. Close the window or Ctrl-C to stop."
        )

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
    print(
        f"[act-eval] env=FrankaSkinHybridObstacleConfig "
        f"samples_per_house={eval_cfg.task_sampler_config.samples_per_house}, "
        f"house_inds={eval_cfg.task_sampler_config.house_inds}, "
        f"task_horizon={eval_cfg.task_horizon}"
    )

    global _WANDB_RUN, _ROLLOUT_INDEX
    if args.use_wandb:
        if wandb is None:
            raise RuntimeError("--use_wandb passed but wandb is not installed.")
        run_name = args.wandb_run_name or f"eval_act_obstacle_{int(time.time())}"
        _WANDB_RUN = wandb.init(
            project=args.wandb_project,
            name=run_name,
            entity=args.wandb_entity,
            group=args.wandb_group,
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
                "env_config_class": "FrankaSkinHybridObstacleConfig",
                "house_ind": args.house_ind,
            },
            tags=["act", "obstacle_baseline", "eval",
                  "temp_agg_off" if pc.temp_agg_off else "temp_agg_on"],
        )
        _ROLLOUT_INDEX = 0
        print(f"[act-eval] wandb run: {_WANDB_RUN.url}  (group={args.wandb_group})")

    try:
        runner = ParallelRolloutRunner(eval_cfg)
        success, total = runner.run()
        rate = (success / total) if total > 0 else 0.0
        print(f"[act-eval] success {success}/{total}  ({rate*100:.1f}%)")

        if _WANDB_RUN is not None:
            _WANDB_RUN.log(
                {
                    "eval/success": int(success),
                    "eval/total": int(total),
                    "eval/success_rate": float(rate),
                }
            )
            _WANDB_RUN.summary["success"] = int(success)
            _WANDB_RUN.summary["total"] = int(total)
            _WANDB_RUN.summary["success_rate"] = float(rate)
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
