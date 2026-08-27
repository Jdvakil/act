"""In-env ACT / PACT eval on the coauthor pick-and-place corridor.

Training data: HF `Lundii/pact_place_corridor_v5` (wrist RGB + 40-sensor skin).
The recovered rows record molmospaces `1cbb180` but the scene XML version is
`pact_place_corridor_v2`, which first appears at `977acd6` together with
`FrankaSkinHybridWristOnlyCameraSystem`. Those pieces live on
`origin/experiment/pact-vs-act-remediation-v2`, not on this checkout's
molmospaces `main`. Point PYTHONPATH / --molmospaces_root at a worktree of
`977acd6`.

    git -C submodules/molmospaces worktree add \\
        /home/jaydv/code/molmospaces-pact-place \\
        977acd6719a8c05b688d3e70da356d61dd32d259

    cd submodules/act
    PYTHONPATH="/home/jaydv/code/molmospaces-pact-place:$PWD:$PYTHONPATH" \\
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \\
    python eval_act_place_corridor.py \\
        --ckpt_dir ckpts/pact_place_corridor_v5/<run> \\
        --output_dir /home/jaydv/code/prox_learning/eval_output/place_corridor_<arm> \\
        --num_rollouts 20 --chunk_size 50 --temp_agg_off --task_horizon 800

Default is metrics-only: no MP4/HDF5, policy loaded once, vanilla skips the
40-sensor 60 Hz depth stack. Start with n=20, then n=50. Kill-safe progress
is `episodes.jsonl` next to `eval_summary.json`.

Never `imitate_episodes.py --eval` on a PACT ckpt.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_LIVE_RENDER = any(a in ("--live", "--render", "--viewer") for a in sys.argv[1:])
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("PACT_CONTACT_AUDIT_SUMMARY_ONLY", "1")
if not _LIVE_RENDER:
    os.environ.pop("DISPLAY", None)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ACT_DIR = Path(__file__).resolve().parent
_DEFAULT_WORKTREE = Path("/home/jaydv/code/molmospaces-pact-place")


def _molmospaces_root_from_argv() -> Path:
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--molmospaces_root" and i + 1 < len(args):
            return Path(args[i + 1]).resolve()
        if a.startswith("--molmospaces_root="):
            return Path(a.split("=", 1)[1]).resolve()
    env = os.environ.get("MOLMOSPACES_PACT_PLACE")
    return Path(env).resolve() if env else _DEFAULT_WORKTREE


_MOLMO_ROOT = _molmospaces_root_from_argv()
if not (_MOLMO_ROOT / "molmo_spaces").is_dir():
    raise SystemExit(
        f"[act-eval-place] molmospaces worktree missing at {_MOLMO_ROOT}.\n"
        "  git -C /home/jaydv/code/prox_learning/submodules/molmospaces worktree add \\\n"
        f"      {_MOLMO_ROOT} 977acd6719a8c05b688d3e70da356d61dd32d259"
    )
sys.path.insert(0, str(_MOLMO_ROOT))
if str(_ACT_DIR) not in sys.path:
    sys.path.insert(0, str(_ACT_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
import time
from pathlib import Path as _Path  # noqa: E402

from eval_act_obstacle import (  # noqa: E402
    ACTInferencePolicy,
    ACTPolicyConfig,
    _EPISODE_METRICS,
)
from utils import set_seed  # noqa: E402

from molmo_spaces.configs.task_configs import PickAndPlaceTaskConfig  # noqa: E402
from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (  # noqa: E402
    FrankaSkinPACTCollisionCorridorConfig,
)
from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner  # noqa: E402
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR  # noqa: E402
from molmo_spaces.tasks.enclosure_reach import PactPlaceCorridorV2Sampler  # noqa: E402
from molmo_spaces.tasks.pact_place_contact_audit import PactPlaceContactAudit  # noqa: E402
from molmo_spaces.tasks.pick_and_place_task import PickAndPlaceTask  # noqa: E402

try:
    import wandb  # type: ignore
except ImportError:
    wandb = None

_METRICS_JSONL: _Path | None = None


def _install_metrics_only_hooks() -> None:
    """Do not keep 800-step RGB/depth histories or write MP4/HDF5.

    The datagen pipeline otherwise retains every episode until the house
    finishes, which OOM-kills a 50-ep eval on a 62 GB box.
    """
    import molmo_spaces.data_generation.pipeline as pipeline
    from molmo_spaces.tasks.task import BaseMujocoTask

    def _skip_save(worker_logger, house_raw_histories, *args, **kwargs):
        n = len(house_raw_histories or [])
        worker_logger.info(
            f"[act-eval-place] skip trajectory/video save ({n} episodes, metrics-only)"
        )
        if house_raw_histories is not None:
            for ep in house_raw_histories:
                if isinstance(ep, dict):
                    ep.pop("sensor_suite", None)
                    ep.pop("history", None)
            house_raw_histories.clear()
        import gc

        gc.collect()

    def _tiny_history(self):
        # Pipeline keeps house_raw_histories until the house ends and stores
        # task.sensor_suite on every episode. Drop the suite here so 50 eval
        # episodes cannot pin 50 copies of 41 cameras.
        if getattr(self, "_sensor_suite", None) is not None:
            self._sensor_suite = None
        return {
            "observations": [],
            "rewards": [],
            "terminals": [],
            "truncateds": [],
            "successes": [],
            "actions": [],
            "obs_scene": {"collision_metrics": self._summarize_collisions()},
        }

    _orig_cache = BaseMujocoTask.get_and_cache_all_step_information

    def _cache_drop_frames(self):
        out = _orig_cache(self)
        if self.observation_cache:
            n_batch = len(self.observation_cache[-1])
            self.observation_cache[-1] = [{} for _ in range(n_batch)]
        return out

    pipeline.save_house_trajectories = _skip_save
    BaseMujocoTask.get_history = _tiny_history
    BaseMujocoTask.get_and_cache_all_step_information = _cache_drop_frames


def _configure_eval_cameras(eval_cfg, *, need_skin: bool) -> None:
    """Vanilla: wrist RGB only. PACT: skin at policy rate, not 60 Hz substeps."""
    eval_cfg.proximity_sensor_period_ms = 0.0
    cams = []
    for cam in list(eval_cfg.camera_config.cameras):
        if (not need_skin) and getattr(cam, "is_proximity_sensor", False):
            continue
        if getattr(cam, "name", None) == "wrist_camera":
            update = {"record_depth": False}
            if hasattr(cam, "model_copy"):
                cam = cam.model_copy(update=update)
            elif hasattr(cam, "copy"):
                cam = cam.copy(update=update)
            else:
                cam.record_depth = False
        cams.append(cam)
    eval_cfg.camera_config.cameras = cams
    n_prox = sum(1 for c in cams if getattr(c, "is_proximity_sensor", False))
    print(
        f"[act-eval-place] cameras={len(cams)} proximity={n_prox} "
        f"period_ms={eval_cfg.proximity_sensor_period_ms} (0=policy-rate)"
    )


class ACTPlaceCorridorEvalConfig(FrankaSkinPACTCollisionCorridorConfig):
    """Wrist-only hybrid-skin pick-and-place corridor; ACT policy swapped in at runtime."""

    policy_config: ACTPolicyConfig = ACTPolicyConfig()
    task_type: str = "pick_and_place"
    task_config: PickAndPlaceTaskConfig = PickAndPlaceTaskConfig(task_cls=PickAndPlaceTask)
    task_horizon: int | None = 800
    viz_sensor_rgb: bool = False
    filter_for_successful_trajectories: bool = False
    use_wandb: bool = False
    num_workers: int = 1
    save_videos: bool = False
    use_passive_viewer: bool = False
    output_dir: _Path = ASSETS_DIR / "datagen" / "act_place_corridor_eval"

    @property
    def tag(self) -> str:
        return "act_place_corridor_eval"


class _PlaceEvalRunner(ParallelRolloutRunner):
    """Attach the place contact audit so ACT rollouts still score hazard-bar hits."""

    @staticmethod
    def run_single_rollout(episode_seed, task, policy, **kwargs):
        audit = PactPlaceContactAudit()
        task._contact_audit_hook = audit
        success = ParallelRolloutRunner.run_single_rollout(
            episode_seed=episode_seed, task=task, policy=policy, **kwargs
        )
        try:
            _record_place_metric(task, bool(success), audit.summary())
        except Exception as e:
            print(f"[act-eval-place] contact metric capture failed: {e}")
        return success


def _record_place_metric(task, success: bool, audit: dict) -> None:
    frames = audit.get("frames_with_contact") or {}
    totals = audit.get("contact_class_totals") or {}
    bar_frames = int(frames.get("hazard_bar") or 0)
    other_frames = int(frames.get("other_environment") or 0)
    clutter_frames = int(frames.get("clutter") or 0)
    rec = {
        "episode_idx": len(_EPISODE_METRICS),
        "success": int(success),
        "hit_bar": int(bar_frames > 0),
        "bar_contact_frames": bar_frames,
        "other_environment_frames": other_frames,
        "clutter_frames": clutter_frames,
        "collision_free": int(bool(audit.get("collision_free"))),
        "intrusion_side": str((getattr(task, "scene_params", {}) or {}).get("pact_intrusion_side") or ""),
        "contact_class_totals": totals,
        "first_contact_step": audit.get("first_contact_step") or {},
    }
    _EPISODE_METRICS.append(rec)
    print(
        f"[act-eval-place] ep{rec['episode_idx']:03d} success={success} "
        f"hit_bar={rec['hit_bar']} bar_frames={bar_frames} "
        f"other={other_frames} clutter={clutter_frames} "
        f"collision_free={rec['collision_free']} side={rec['intrusion_side'] or '-'}"
    )
    if _METRICS_JSONL is not None:
        with _METRICS_JSONL.open("a") as handle:
            handle.write(json.dumps(rec) + "\n")


def _summarize_place_metrics() -> dict | None:
    if not _EPISODE_METRICS:
        return None
    n = len(_EPISODE_METRICS)
    bar_hits = sum(int(m["hit_bar"]) for m in _EPISODE_METRICS)
    successes = sum(int(m["success"]) for m in _EPISODE_METRICS)
    collision_free = sum(int(m["collision_free"]) for m in _EPISODE_METRICS)
    return {
        "episodes": n,
        "success": successes,
        "success_rate": successes / n,
        "bar_hits": bar_hits,
        "bar_hit_rate": bar_hits / n,
        "collision_free": collision_free,
        "collision_free_rate": collision_free / n,
        "collision_rate": 1.0 - (collision_free / n),
        "episodes_detail": list(_EPISODE_METRICS),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--ckpt_name", default="policy_best.ckpt")
    p.add_argument(
        "--output_dir",
        default="/home/jaydv/code/prox_learning/eval_output/pact_place_corridor_v5",
    )
    p.add_argument("--num_rollouts", type=int, default=50)
    p.add_argument("--house_ind", type=int, default=1)
    p.add_argument("--task_horizon", type=int, default=800)
    p.add_argument("--chunk_size", type=int, default=50)
    p.add_argument("--kl_weight", type=int, default=10)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--dim_feedforward", type=int, default=3200)
    p.add_argument("--image_h", type=int, default=240)
    p.add_argument("--image_w", type=int, default=320)
    p.add_argument("--temp_agg_off", action="store_true")
    p.add_argument("--temp_agg_m", type=float, default=0.01)
    p.add_argument("--eval_blur_sigma", type=float, default=0.0)
    p.add_argument("--use_proximity", action="store_true")
    p.add_argument("--prox_encoder_ckpt", type=str, default="")
    p.add_argument("--prox_feature", type=str, default="raw")
    p.add_argument("--prox_tokens_per_sensor", type=int, default=8)
    p.add_argument("--prox_layout", type=str, default="per_sensor")
    p.add_argument(
        "--save_trajectories",
        action="store_true",
        help="Keep datagen MP4/HDF5 (slow, OOM-prone). Default is metrics-only.",
    )
    p.add_argument("--live", "--render", "--viewer", dest="live", action="store_true")
    p.add_argument(
        "--molmospaces_root",
        type=str,
        default=str(_DEFAULT_WORKTREE),
        help="Worktree of molmospaces 977acd6 (pact_place_corridor_v2 env).",
    )
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="act-place-corridor-eval")
    p.add_argument("--wandb_run_name", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.temp_agg_off:
        print(
            "[act-eval-place] WARNING: --temp_agg_off not set. Open-loop chunking "
            "is the valid PACT path; temp-agg-on almost ignores live skin."
        )

    eval_cfg = ACTPlaceCorridorEvalConfig()
    eval_cfg.task_horizon = args.task_horizon
    eval_cfg.output_dir = Path(args.output_dir).resolve()
    eval_cfg.output_dir.mkdir(parents=True, exist_ok=True)

    import molmo_spaces as _ms

    scenes = Path(_ms.__file__).resolve().parent / "data_generation" / "custom_scenes"
    xml = scenes / "pact_place_corridor_v2.xml"
    if not xml.is_file():
        raise SystemExit(f"[act-eval-place] missing scene XML {xml} (wrong molmospaces root?)")
    eval_cfg.task_sampler_config.task_sampler_class = PactPlaceCorridorV2Sampler
    eval_cfg.task_sampler_config.scene_xml_paths = [str(xml)] * 2
    eval_cfg.task_sampler_config.samples_per_house = args.num_rollouts
    eval_cfg.task_sampler_config.house_inds = [args.house_ind]

    live = args.live
    if live and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print("[act-eval-place] --live requested but no display; headless")
        live = False
    eval_cfg.use_passive_viewer = live
    if live:
        eval_cfg.num_workers = 1

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
    pc.eval_blur_sigma = float(args.eval_blur_sigma)
    pc.camera_names = ("wrist_camera",)

    prox_cfg_path = Path(pc.ckpt_dir) / "prox_config.json"
    if prox_cfg_path.exists():
        pcfg = json.loads(prox_cfg_path.read_text())
        pc.use_proximity = True
        pc.prox_feature = pcfg.get("prox_feature", "raw")
        pc.prox_layout = pcfg.get("prox_layout", "per_sensor")
        pc.prox_pool = pcfg.get("prox_pool", "min")
        pc.prox_tokens_per_sensor = int(pcfg.get("prox_tokens_per_sensor", 8))
        pc.prox_encoder_ckpt = args.prox_encoder_ckpt or pcfg.get("prox_encoder_ckpt", "")
        print(
            f"[act-eval-place] PACT ckpt -> proximity ON "
            f"(feature={pc.prox_feature}, layout={pc.prox_layout}, "
            f"K={pc.prox_tokens_per_sensor}, pool={pc.prox_pool})"
        )
    elif args.use_proximity:
        pc.use_proximity = True
        pc.prox_feature = args.prox_feature
        pc.prox_layout = args.prox_layout
        pc.prox_tokens_per_sensor = args.prox_tokens_per_sensor
        pc.prox_encoder_ckpt = args.prox_encoder_ckpt
        print("[act-eval-place] proximity FORCED ON via CLI")

    global _METRICS_JSONL
    _METRICS_JSONL = eval_cfg.output_dir / "episodes.jsonl"
    _configure_eval_cameras(eval_cfg, need_skin=bool(pc.use_proximity))
    if not args.save_trajectories:
        _install_metrics_only_hooks()
        print("[act-eval-place] metrics-only eval (no MP4/HDF5, policy loaded once)")

    eval_cfg.save_config()
    print(
        f"[act-eval-place] molmospaces={_MOLMO_ROOT} "
        f"sampler={eval_cfg.task_sampler_config.task_sampler_class.__name__} "
        f"xml={xml.name} cameras={pc.camera_names} "
        f"horizon={eval_cfg.task_horizon} n={args.num_rollouts}"
    )
    print(f"[act-eval-place] writing {eval_cfg.output_dir}")

    os.environ.pop("WANDB_RUN_NAME", None)
    os.environ.pop("WANDB_PROJECT_NAME", None)

    if args.use_wandb:
        if wandb is None:
            raise RuntimeError("wandb not installed")
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"eval_place_corridor_{int(time.time())}",
            config={"ckpt_dir": pc.ckpt_dir, "num_rollouts": args.num_rollouts},
        )

    runner = _PlaceEvalRunner(eval_cfg)
    policy = ACTInferencePolicy(eval_cfg)
    policy.prepare_model()
    success, total = runner.run(preloaded_policy=policy)
    rate = (success / total) if total else 0.0
    print(f"[act-eval-place] success {success}/{total}  ({rate*100:.1f}%)")
    collision = _summarize_place_metrics()
    if collision is not None:
        print(
            f"[act-eval-place] bar_hit {collision['bar_hits']}/{collision['episodes']} "
            f"({collision['bar_hit_rate']*100:.1f}%)  "
            f"collision_free {collision['collision_free']}/{collision['episodes']} "
            f"({collision['collision_free_rate']*100:.1f}%)"
        )

    summary = {
        "ckpt_dir": pc.ckpt_dir,
        "ckpt_name": pc.ckpt_name,
        "task": "pact_place_corridor_v5",
        "molmospaces_root": str(_MOLMO_ROOT),
        "task_sampler_class": eval_cfg.task_sampler_config.task_sampler_class.__name__,
        "scene_xml": str(xml),
        "camera_names": list(pc.camera_names),
        "num_rollouts": args.num_rollouts,
        "task_horizon": args.task_horizon,
        "chunk_size": pc.chunk_size,
        "temp_agg_off": pc.temp_agg_off,
        "use_proximity": pc.use_proximity,
        "success": int(success),
        "total": int(total),
        "success_rate": float(rate),
        "collision": collision,
    }
    out = eval_cfg.output_dir / "eval_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[act-eval-place] wrote {out}")


if __name__ == "__main__":
    main()
