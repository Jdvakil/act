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
import json
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
# Per-episode metrics collected during runner.run() by _ObstacleEvalRunner. Valid only
# for the single-worker, in-process eval path (num_workers=1), which is how this harness
# always runs — the rollout shares this process, so the live task object and these
# globals are reachable. Each entry is one dict (see _record_episode_collision_metric).
_EPISODE_METRICS: list[dict] = []
# Strict-safety criterion. When True (set in main from --end_on_collision), each rollout
# ends at the first arm<->obstacle contact and the reported success is collision-as-failure.
_END_ON_COLLISION = False


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
import prox_cvae
from prox_cvae import ProxCVAEEncoder, stack_obs_proximity

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
        self._prox_encoder = None  # P+ACT frozen Safety-CVAE feature extractor

    def reset(self) -> None:
        # Per-episode wandb logging (index, length, collisions) is owned by
        # _ObstacleEvalRunner.run_single_rollout, which fires once per finished episode
        # and is the sole writer of _ROLLOUT_INDEX. The policy is rebuilt every episode
        # (pipeline setup_policy), so accumulating episode state here is unreliable.
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
        # P+ACT: build the SAME frozen Safety-CVAE feature extractor used at train time
        # and switch on the proximity tokens (n_proximity_sensors=1, K, feat_dim). The
        # values come from the ckpt's prox_config.json (set in main), so train/eval match.
        if pc.use_proximity:
            ckpt = pc.prox_encoder_ckpt or prox_cvae.DEFAULT_CKPT
            self._prox_encoder = ProxCVAEEncoder(
                ckpt, feature=pc.prox_feature, device="cuda",
                layout=getattr(pc, "prox_layout", "global"),
                tokens_per_sensor=int(pc.prox_tokens_per_sensor),
            )
            policy_config["n_proximity_sensors"] = self._prox_encoder.n_act_sensors
            policy_config["prox_tokens_per_sensor"] = self._prox_encoder.tokens_per_sensor
            policy_config["prox_feat_dim"] = self._prox_encoder.act_feat_dim
            self._prox_pool = getattr(pc, "prox_pool", "mean")
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

        # --temp_agg_off = standard ACT open-loop chunking: query the policy once,
        # execute the WHOLE chunk, re-query only when it is exhausted. Re-querying
        # every step and executing chunk[0] is degenerate — action[0] is nearly a
        # copy of the current qpos (the easiest thing for the L1 loss to learn), so
        # per-step chunk[0] execution converges to a fixed point where the arm
        # freezes mid-approach (observed: 0/10 across all arms, arm stalls ~30 cm
        # short of the object for the rest of the episode).
        if pc.temp_agg_off and self._pending_chunks:
            start, chunk = self._pending_chunks[0]
            k = self._step - start
            if 0 <= k < len(chunk):
                return chunk[k]

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

        # Eval-time visual corruption. Blurs the camera frames the policy sees while
        # leaving qpos and the 40-sensor skin depths untouched, which is the point: it
        # degrades exactly one modality. Kernel and sigma convention match
        # imitate_episodes.blur_images so a policy trained at constant sigma S and
        # evaluated at --eval_blur_sigma S sees the same corruption both times.
        # Applied on the 0-1 tensor, before ACTPolicy's ImageNet Normalize (the blur
        # commutes with a per-channel affine). Default 0.0 = sharp = old behaviour.
        if pc.eval_blur_sigma >= 0.1:
            from torchvision.transforms.functional import gaussian_blur
            _b, _k, _c, _h, _w = image_t.shape
            _kernel = 2 * int(np.ceil(3 * pc.eval_blur_sigma)) + 1
            image_t = gaussian_blur(
                image_t.reshape(_b * _k, _c, _h, _w),
                kernel_size=_kernel, sigma=float(pc.eval_blur_sigma),
            ).reshape(_b, _k, _c, _h, _w)
            if self._step == 0:
                print(f"[act-eval] EVAL BLUR ON: sigma={pc.eval_blur_sigma} "
                      f"kernel={_kernel} on {pc.camera_names} (skin/qpos untouched)")

        # P+ACT: stack the live 40-sensor skin depths (CVAE meta order) and run the frozen
        # extractor to get the conditioning feature. Raw meters -> extractor featurizes.
        proximity_positions = None
        if self._prox_encoder is not None:
            prox_np = stack_obs_proximity(
                obs, self._prox_encoder.sensor_order, pool=getattr(self, "_prox_pool", "mean")
            )  # (40,8,8)
            if self._step == 0:
                print(f"[act-eval] proximity ON | {prox_np.shape} "
                      f"min={prox_np.min():.3f}m max={prox_np.max():.3f}m")
            prox_t = torch.from_numpy(prox_np).float().cuda().unsqueeze(0)  # (1,40,8,8)
            proximity_positions = self._prox_encoder(prox_t)               # (1,1,feat_dim)

        with torch.no_grad():
            a_hat = self._policy(qpos_t, image_t, proximity_positions=proximity_positions)
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
    # Gaussian blur sigma (pixels) applied to camera frames at EVAL time only.
    # 0.0 = sharp. Independent of the training-time --blur_sigma0/--blur_mode.
    eval_blur_sigma: float = 0.0
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

    # P+ACT (PACT) proximity fusion. Auto-populated in main() from the ckpt's
    # prox_config.json; vanilla ckpts have no such file -> use_proximity stays False.
    use_proximity: bool = False
    prox_encoder_ckpt: str = ""
    prox_feature: str = "raw"
    prox_layout: str = "global"
    prox_pool: str = "mean"
    prox_tokens_per_sensor: int = 8


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
    # The skin-sensor high-res RGB/depth visualization (ProximityVizRGBSensor /
    # ProximityVizDepthSensor) is inherited as True from the FrankaSkinHybrid* chain and
    # renders all 40 proximity sensors at 256x256 RGB + 256x256 depth-turbo on EVERY policy
    # step. The vision-only ACT policy never consumes those keys (it reads only
    # exo_camera_1 + wrist_camera RGB + qpos), yet the data-generation pipeline retains each
    # episode's full observation history in RAM until the whole house finishes saving --
    # ~3 GB/episode of purely cosmetic skin frames, which OOM-kills any multi-episode eval.
    # Disabling it for eval leaves the policy inputs, success judging, and the arm<->obstacle
    # collision metric byte-for-byte identical while keeping memory flat and rollouts faster.
    viz_sensor_rgb: bool = False
    use_passive_viewer: bool = False
    # Strict-safety eval criterion (--end_on_collision). When True, PickTask demotes any
    # episode with an arm<->obstacle collision to a FAILURE and ends it the moment of first
    # contact (see pick_task.is_terminal / get_info). The reported success rate then becomes
    # the collision-as-failure ("strict") rate. Default False keeps the standard protocol;
    # note _summarize_collision_metrics ALSO reports strict_success_rate in either mode, so a
    # normal run yields both the raw and the hedged numbers without re-running.
    end_on_collision: bool = False
    output_dir: Path = ASSETS_DIR / "datagen" / "act_obstacle_eval"

    @property
    def tag(self) -> str:
        return "act_obstacle_eval"


# ----------------------------------------------------------------------
# Per-cell eval (--eval_cell): score the SAME checkpoint on one obstacle cell per
# invocation by swapping the inherited ObstacleFumehoodPickSampler for
# InvisibleObstacleFumehoodPickCheckSampler and pinning its episode-mix class attrs.
# ----------------------------------------------------------------------
# cell -> (OBSTACLE_P, INVIS_P) forced on the sampler class.
_EVAL_CELL_PROBS = {
    "visible": (1.0, 0.0),    # bar present on every episode, rendered to RGB as usual
    "invisible": (1.0, 1.0),  # bar present on every episode, hidden from RGB (skin-only)
    "free": (0.0, 0.0),       # bar never present; same sampler keeps the object-placement
                              # distribution identical across the three cells
}


def _apply_eval_cell(eval_cfg, cell: str) -> None:
    """Pin the eval task mix to one obstacle cell (see _EVAL_CELL_PROBS).

    Imported lazily so this script keeps working (without --eval_cell) against an older
    molmospaces checkout that predates the invisible-bar sampler. The probabilities are
    forced as CLASS attributes (the sampler reads them via self.*): eval runs one cell
    per process, and mutating the real importable class -- rather than materializing a
    dynamic subclass -- keeps eval_cfg picklable for save_config().
    """
    try:
        from molmo_spaces.tasks.enclosure_reach import (
            InvisibleObstacleFumehoodPickCheckSampler as _CellSampler,
        )
    except ImportError as e:
        raise SystemExit(
            f"--eval_cell {cell!r} requires InvisibleObstacleFumehoodPickCheckSampler in "
            f"molmo_spaces.tasks.enclosure_reach; this molmospaces checkout does not have "
            f"it ({e}). Update the submodule or drop --eval_cell."
        )
    obstacle_p, invis_p = _EVAL_CELL_PROBS[cell]
    _CellSampler.OBSTACLE_P = obstacle_p
    _CellSampler.INVIS_P = invis_p
    eval_cfg.task_sampler_config.task_sampler_class = _CellSampler
    print(
        f"[act-eval] eval_cell={cell} -> sampler=InvisibleObstacleFumehoodPickCheckSampler "
        f"(OBSTACLE_P={obstacle_p}, INVIS_P={invis_p})"
    )


# ----------------------------------------------------------------------
# Collision metric capture.
#
# PickTask accumulates a per-step arm<->obstacle penetrating-contact count on the live
# task object (`task._obstacle_diag`: {episode_step -> # distinct obstacle bodies the
# arm is wedged against}; see molmospaces/.../tasks/pick_task.py). The molmospaces
# pipeline only returns (success, total), so we subclass ParallelRolloutRunner and
# override its `run_single_rollout` hook to read that diagnostic off the task the moment
# each episode ends, turning it into a per-episode collision metric we can log.
#
# This works because eval runs single-worker (num_workers=1), which ParallelRolloutRunner
# executes IN THE MAIN PROCESS, so the same `task` object and this module's globals are
# in scope. run_single_rollout is dispatched as `runner_class.run_single_rollout(...)`
# with runner_class=type(self), so the override below is the one that runs.
# ----------------------------------------------------------------------
class _ObstacleEvalRunner(ParallelRolloutRunner):
    """Runner that records each episode's arm<->obstacle contact metric after rollout."""

    @staticmethod
    def run_single_rollout(episode_seed, task, policy, **kwargs):
        # Strict criterion: end the episode the moment the arm first penetrates an obstacle
        # body. task._obstacle_diag[step] (>0 => contact) is populated during task.step();
        # shadowing is_done on this per-rollout task instance makes the pipeline loop
        # (`while not task.is_done()`) exit ~1 step after first contact. Verified against
        # pipeline.py:732 + pick_task.py:225. No submodule edit needed.
        if _END_ON_COLLISION:
            _orig_is_done = task.is_done
            def _is_done_or_collided():
                diag = getattr(task, "_obstacle_diag", {}) or {}
                if any(v > 0 for v in diag.values()):
                    return True
                return _orig_is_done()
            task.is_done = _is_done_or_collided
        success = ParallelRolloutRunner.run_single_rollout(
            episode_seed=episode_seed, task=task, policy=policy, **kwargs
        )
        collided = any(v > 0 for v in (getattr(task, "_obstacle_diag", {}) or {}).values())
        try:
            _record_episode_collision_metric(task, bool(success))
        except Exception as e:  # metric capture must never break a rollout
            print(f"[act-eval] collision metric capture failed: {e}")
        # Under the strict criterion the reported success rate is collision-as-failure:
        # a grasp that ever touched an obstacle does not count.
        if _END_ON_COLLISION:
            return bool(success) and not collided
        return success


def _record_episode_collision_metric(task, success: bool) -> None:
    """Read task._obstacle_diag for the just-finished episode -> per-episode record."""
    global _ROLLOUT_INDEX
    diag = dict(getattr(task, "_obstacle_diag", {}) or {})
    length = len(diag)
    contact_steps = sum(1 for v in diag.values() if v > 0)
    peak = max(diag.values()) if diag else 0
    first = next((s for s in sorted(diag) if diag[s] > 0), None)
    rec = {
        "episode_idx": _ROLLOUT_INDEX,
        "success": int(success),
        "length": int(length),
        "obstacle_contact_steps": int(contact_steps),
        "obstacle_contact_fraction": (contact_steps / length) if length else 0.0,
        "obstacle_peak_contacts": int(peak),
        "obstacle_first_contact_step": (-1 if first is None else int(first)),
        "obstacle_contact_free": int(contact_steps == 0),
    }
    _EPISODE_METRICS.append(rec)
    print(
        f"[act-eval] ep{_ROLLOUT_INDEX:03d} success={success} "
        f"obstacle_contact_steps={contact_steps}/{length} "
        f"peak={peak} first_contact_step={first}"
    )
    # Per-episode wandb point. Explicit step= keeps the series monotonic and decoupled
    # from any auto-incremented logs; the final aggregate is logged at step=N afterward.
    if _WANDB_RUN is not None:
        _WANDB_RUN.log(
            {
                "episode/success": rec["success"],
                "episode/length": rec["length"],
                "episode/obstacle_contact_steps": rec["obstacle_contact_steps"],
                "episode/obstacle_contact_fraction": rec["obstacle_contact_fraction"],
                "episode/obstacle_peak_contacts": rec["obstacle_peak_contacts"],
                "episode/obstacle_first_contact_step": rec["obstacle_first_contact_step"],
                "episode/obstacle_contact_free": rec["obstacle_contact_free"],
            },
            step=_ROLLOUT_INDEX,
        )
    _ROLLOUT_INDEX += 1


def _summarize_collision_metrics() -> dict | None:
    """Aggregate the per-episode collision records into eval-level scalars."""
    n = len(_EPISODE_METRICS)
    if n == 0:
        return None

    def mean(xs: list[float]) -> float:
        return (sum(xs) / len(xs)) if xs else 0.0

    succ = [m for m in _EPISODE_METRICS if m["success"]]
    fail = [m for m in _EPISODE_METRICS if not m["success"]]
    # Strict (collision-as-failure) success: grasped+lifted AND never touched an obstacle.
    # Computed in BOTH modes -> a normal run reports the hedged number too. Under
    # --end_on_collision this equals the raw success_rate (collisions already fail + truncate).
    collision_rate = mean([1 - m["obstacle_contact_free"] for m in _EPISODE_METRICS])
    strict_success = sum(1 for m in _EPISODE_METRICS if m["success"] and m["obstacle_contact_free"])
    return {
        "episodes": n,
        "mean_contact_steps": mean([m["obstacle_contact_steps"] for m in _EPISODE_METRICS]),
        "mean_contact_fraction": mean([m["obstacle_contact_fraction"] for m in _EPISODE_METRICS]),
        "mean_peak_contacts": mean([m["obstacle_peak_contacts"] for m in _EPISODE_METRICS]),
        "contact_free_rate": mean([m["obstacle_contact_free"] for m in _EPISODE_METRICS]),
        "collision_rate": collision_rate,
        "strict_success": strict_success,
        "strict_success_rate": strict_success / n,
        "mean_contact_steps_success": mean([m["obstacle_contact_steps"] for m in succ]),
        "mean_contact_steps_fail": mean([m["obstacle_contact_steps"] for m in fail]),
    }


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
    p.add_argument("--eval_blur_sigma", type=float, default=0.0,
                   help="Gaussian blur sigma (pixels) applied to the camera frames at "
                        "EVAL time. Corrupts vision only; the proximity skin and qpos "
                        "are untouched. 0 = sharp. Use to sweep graceful degradation "
                        "of a fixed checkpoint (paired: same weights, same seeds).")
    p.add_argument(
        "--end_on_collision", action="store_true",
        help="Strict-safety criterion: any arm<->obstacle collision counts as a FAILURE and "
             "ends the episode on first contact. Reported success rate becomes the strict "
             "(collision-as-failure) rate. Default off.",
    )
    p.add_argument(
        "--eval_cell", choices=("visible", "invisible", "free"), default=None,
        help="Pin every rollout to one obstacle cell via "
             "InvisibleObstacleFumehoodPickCheckSampler: 'visible' = bar always present and "
             "rendered to RGB; 'invisible' = bar always present but hidden from the RGB "
             "cameras (skin-only); 'free' = bar never present. Default: flag absent keeps "
             "the inherited ObstacleFumehoodPickSampler mix (~75%% visible bar), i.e. the "
             "previous behavior, and never imports the invisible-bar sampler.",
    )
    # P+ACT (PACT): usually auto-detected from <ckpt_dir>/prox_config.json, so these are
    # only needed to force/override proximity on a ckpt without that file.
    p.add_argument("--use_proximity", action="store_true",
                   help="Force proximity fusion ON (normally auto-detected from prox_config.json).")
    p.add_argument("--prox_encoder_ckpt", type=str, default="",
                   help="Safety-CVAE dir (default: assets/safety/cvae_v3).")
    p.add_argument("--prox_feature", type=str, default="trunk",
                   choices=("trunk", "delta", "raw"))
    p.add_argument("--prox_tokens_per_sensor", type=int, default=8)
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
    eval_cfg.end_on_collision = bool(args.end_on_collision)
    global _END_ON_COLLISION
    _END_ON_COLLISION = eval_cfg.end_on_collision
    eval_cfg.output_dir = Path(args.output_dir).resolve()
    eval_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if eval_cfg.end_on_collision:
        print(
            "[act-eval] STRICT mode ON (--end_on_collision): any arm<->obstacle collision "
            "= FAILURE, episode ends on first contact. Reported success = strict rate."
        )

    # Drive N rollouts of the single red-cup task from one process.
    eval_cfg.task_sampler_config.samples_per_house = args.num_rollouts
    eval_cfg.task_sampler_config.house_inds = [args.house_ind]
    if args.eval_cell:
        _apply_eval_cell(eval_cfg, args.eval_cell)

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
    pc.eval_blur_sigma = float(args.eval_blur_sigma)
    if pc.eval_blur_sigma >= 0.1:
        print(f"[act-eval] eval-time camera blur sigma={pc.eval_blur_sigma} "
              f"(vision degraded, proximity + qpos untouched)")
    pc.temp_agg_m = args.temp_agg_m

    # P+ACT: a PACT checkpoint carries prox_config.json (written by imitate_episodes.py).
    # Prefer it so eval rebuilds the EXACT train-time extractor + token layout; fall back
    # to CLI flags only when forcing proximity on a ckpt without it. Vanilla ckpts have no
    # such file -> proximity stays off and this is a no-op.
    prox_cfg_path = Path(pc.ckpt_dir) / "prox_config.json"
    if prox_cfg_path.exists():
        pcfg = json.loads(prox_cfg_path.read_text())
        pc.use_proximity = True
        pc.prox_feature = pcfg.get("prox_feature", "trunk")
        pc.prox_layout = pcfg.get("prox_layout", "global")
        pc.prox_pool = pcfg.get("prox_pool", "mean")
        pc.prox_tokens_per_sensor = int(pcfg.get("prox_tokens_per_sensor", 8))
        pc.prox_encoder_ckpt = args.prox_encoder_ckpt or pcfg.get("prox_encoder_ckpt", "")
        print(f"[act-eval] PACT ckpt detected -> proximity ON "
              f"(feature={pc.prox_feature}, layout={pc.prox_layout}, "
              f"K={pc.prox_tokens_per_sensor}, pool={pc.prox_pool})")
    elif args.use_proximity:
        pc.use_proximity = True
        pc.prox_feature = args.prox_feature
        pc.prox_layout = getattr(args, "prox_layout", "global")
        pc.prox_pool = getattr(args, "prox_pool", None) or "mean"
        pc.prox_tokens_per_sensor = args.prox_tokens_per_sensor
        pc.prox_encoder_ckpt = args.prox_encoder_ckpt
        print(f"[act-eval] proximity FORCED ON via CLI "
              f"(feature={pc.prox_feature}, layout={pc.prox_layout}, "
              f"K={pc.prox_tokens_per_sensor})")

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
                "end_on_collision": eval_cfg.end_on_collision,
                "eval_cell": args.eval_cell,
            },
            tags=["act", "obstacle_baseline", "eval",
                  "temp_agg_off" if pc.temp_agg_off else "temp_agg_on",
                  "strict_collision" if eval_cfg.end_on_collision else "standard"]
                 + ([f"cell_{args.eval_cell}"] if args.eval_cell else []),
        )
        _ROLLOUT_INDEX = 0
        print(f"[act-eval] wandb run: {_WANDB_RUN.url}  (group={args.wandb_group})")

    # molmospaces' pipeline starts (and later wandb.finish()-es) its OWN global wandb run
    # when these env vars are set (pipeline.py gates on them, NOT on use_wandb). That run
    # would hijack and prematurely finish the run this harness manages via _WANDB_RUN. We
    # drive wandb ourselves, so drop them to keep the pipeline's internal wandb off.
    os.environ.pop("WANDB_RUN_NAME", None)
    os.environ.pop("WANDB_PROJECT_NAME", None)

    try:
        runner = _ObstacleEvalRunner(eval_cfg)
        success, total = runner.run()
        rate = (success / total) if total > 0 else 0.0
        label = "strict success (collision=fail)" if eval_cfg.end_on_collision else "success"
        print(f"[act-eval] {label} {success}/{total}  ({rate*100:.1f}%)")

        collision = _summarize_collision_metrics()
        if collision is not None:
            print(
                "[act-eval] collision summary: "
                f"collision_rate={collision['collision_rate']:.2f} "
                f"contact_free_rate={collision['contact_free_rate']:.2f} "
                f"mean_contact_steps={collision['mean_contact_steps']:.1f} "
                f"(success={collision['mean_contact_steps_success']:.1f}, "
                f"fail={collision['mean_contact_steps_fail']:.1f}) "
                f"mean_peak={collision['mean_peak_contacts']:.2f}"
            )
            # The hedged headline: grasped+lifted AND never touched an obstacle.
            print(
                f"[act-eval] strict_success (collision=fail) "
                f"{collision['strict_success']}/{collision['episodes']}  "
                f"({collision['strict_success_rate']*100:.1f}%)"
            )

        # Self-describing results file: which checkpoint was scored on which obstacle
        # cell (eval_cell=null means the inherited ~75%-visible-bar mix), plus the
        # headline numbers and the per-episode collision records.
        summary = {
            "ckpt_dir": pc.ckpt_dir,
            "ckpt_name": pc.ckpt_name,
            "eval_cell": args.eval_cell,
            "task_sampler_class": eval_cfg.task_sampler_config.task_sampler_class.__name__,
            "num_rollouts": args.num_rollouts,
            "house_ind": args.house_ind,
            "task_horizon": args.task_horizon,
            "temp_agg_off": pc.temp_agg_off,
            "eval_blur_sigma": pc.eval_blur_sigma,
            "end_on_collision": eval_cfg.end_on_collision,
            "use_proximity": pc.use_proximity,
            "success": int(success),
            "total": int(total),
            "success_rate": float(rate),
            "collision": collision,
            "episodes": _EPISODE_METRICS,
        }
        if args.eval_cell:
            summary["obstacle_p"], summary["invis_p"] = _EVAL_CELL_PROBS[args.eval_cell]
        summary_path = eval_cfg.output_dir / "eval_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"[act-eval] wrote {summary_path}")

        if _WANDB_RUN is not None:
            final_step = len(_EPISODE_METRICS)
            agg = {
                "eval/success": int(success),
                "eval/total": int(total),
                "eval/success_rate": float(rate),
            }
            if collision is not None:
                agg.update({f"eval/{k}": float(v) for k, v in collision.items()})
            # step=final_step sits just past the per-episode points (0..N-1), keeping
            # the wandb step sequence strictly non-decreasing.
            _WANDB_RUN.log(agg, step=final_step)
            for k, v in agg.items():
                _WANDB_RUN.summary[k] = v
            _log_per_episode_table_to_wandb(step=final_step)
            _log_rollout_videos_to_wandb(eval_cfg.output_dir, step=final_step)
    finally:
        if _WANDB_RUN is not None:
            _WANDB_RUN.finish()
            _WANDB_RUN = None


def _log_per_episode_table_to_wandb(step: int | None = None) -> None:
    """Log the full per-episode collision breakdown as a single wandb.Table."""
    if wandb is None or _WANDB_RUN is None or not _EPISODE_METRICS:
        return
    cols = [
        "episode_idx", "success", "length",
        "obstacle_contact_steps", "obstacle_contact_fraction",
        "obstacle_peak_contacts", "obstacle_first_contact_step", "obstacle_contact_free",
    ]
    table = wandb.Table(columns=cols)
    for m in _EPISODE_METRICS:
        table.add_data(*[m[c] for c in cols])
    try:
        _WANDB_RUN.log({"eval/per_episode": table}, step=step)
    except Exception as e:
        print(f"[act-eval] could not log per-episode table: {e}")


def _log_rollout_videos_to_wandb(output_dir: Path, step: int | None = None) -> None:
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
                _WANDB_RUN.log({key: wandb.Video(str(mp4), format="mp4")}, step=step)
                n_logged += 1
            except Exception as e:
                print(f"[act-eval] could not log {mp4.name}: {e}")
    print(f"[act-eval] uploaded {n_logged} rollout videos to wandb")


if __name__ == "__main__":
    main()
