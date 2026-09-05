"""In-env ACT / PACT eval for pact_pick_n_place_v2 (v1011d train).

What this scores
----------------
Same random-house protocol as hallway v5, different env. Drive the trained
policy, write place-success / bar_hit / collision_free / strict
(success ∧ collision-free) / gripper_close_commanded. Val loss is not this.
Floor for a claim is n≈48 (2 per 24-cell) or n=50 pin (one cell — biased).
Never ``imitate_episodes.py --eval``.

Env pin — THIS IS OOD vs the v1011d dump
----------------------------------------
Train rows: ``pact_place_corridor_v10_11d_randomized_clutter``. Six live
clutter bodies (``pact_primitive_cylinder_01``, Plate_10, Plate_22,
Soap_Bottle_11, ``pact_primitive_cylinder_08``, ``pact_primitive_box_09``).
Collect sampler: ``PactPlaceCorridorV1011DRandomizedLayoutSampler`` on
molmospaces ``70dedc0`` (``origin/experiment/pact-vs-act-remediation-v2``).
That class is **not** on ``origin/main``.

This script still samples ``PactPlaceCorridorV1010FourObjectSampler`` from
``origin/main`` (``4bba4cb``): four frozen household objects, slot 01 is
Soap_Bottle_30 not the cylinder, slots 08/09 parked. XML family is the same
hashed ``v10_7_*`` files. The 0/48 results at horizons 800 and 1050 therefore
measure an OOD task. The mismatch is not proof of the cause of every failure;
the success judge still needs positive/negative runtime checks. Do not cite
those JSONs as matched-environment PACT-vs-ACT results.

prox_learning ``7d1ea35`` added ``custom_scenes/pact_place_corridor_v12.xml``.
Hashed runtime XML plus the v5→v3 include chain now live in the same folder.
Eval loads those files. The v12 wrapper is never a sampler path (hash mismatch).

Do not import ``eval_act_place_corridor.py`` from here — that module pins
worktree ``977acd6`` (hallway v2) at import.

    git -C /home/jaydv/code/prox_learning/submodules/molmospaces worktree add \\
        /home/jaydv/code/molmospaces-pact-v1010 origin/main

    cd /home/jaydv/code/prox_learning/submodules/act
    PYTHONPATH="/home/jaydv/code/molmospaces-pact-v1010:$PWD:$PYTHONPATH" \\
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \\
    python eval_act_pact_pick_n_place.py \\
        --ckpt_dir ckpts/pact_pick_n_place_v2/20260903_171108_pact_pick_n_place_v2_v1011d_s0 \\
        --output_dir /home/jaydv/code/prox_learning/eval_output/pact_pick_n_place_v2_v1011d_raw_s0_n2 \\
        --num_rollouts 2 --chunk_size 50 --temp_agg_off --task_horizon 1050

Smoke = n=2, no ``--spread_cells`` (house 1 = F0 left center). Real eval =
``--spread_cells --num_rollouts 48`` (2 per 24 cells). Metrics-only default.
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

from eval_place_v1010_scene import (  # noqa: E402
    DEFAULT_WORKTREE,
    MOLMOSPACES_V1010_SHA,
    N_V1010_CELLS,
    V12_XML,
    assert_v12_wraps_center,
    resolve_v1010_scenes_dir,
    spread_episode_count,
    v1010_cell,
    v1010_scene_paths,
)


def _molmospaces_root_from_argv() -> Path:
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--molmospaces_root" and i + 1 < len(args):
            return Path(args[i + 1]).resolve()
        if a.startswith("--molmospaces_root="):
            return Path(a.split("=", 1)[1]).resolve()
    env = os.environ.get("MOLMOSPACES_PACT_V1010")
    return Path(env).resolve() if env else DEFAULT_WORKTREE


_MOLMO_ROOT = _molmospaces_root_from_argv()
if not (_MOLMO_ROOT / "molmo_spaces").is_dir():
    raise SystemExit(
        f"[act-eval-pact-v2] molmospaces worktree missing at {_MOLMO_ROOT}.\n"
        "  git -C /home/jaydv/code/prox_learning/submodules/molmospaces worktree add \\\n"
        f"      {_MOLMO_ROOT} {MOLMOSPACES_V1010_SHA}\n"
        "  # origin/main pin. Do not point this at molmospaces-pact-place (977acd6)."
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
from eval_place_fast_hooks import (  # noqa: E402
    _configure_eval_cameras,
    _install_chunk_gated_sensors,
    _install_metrics_only_hooks,
    _install_raycast_proximity,
)
from utils import set_seed  # noqa: E402

from molmo_spaces.configs.camera_configs import (  # noqa: E402
    FrankaSkinHybridCameraSystem,
)
from molmo_spaces.data_generation.config.pact_place_datagen_configs import (  # noqa: E402
    FrankaSkinPactPlaceV1010FourObjectConfig,
)
from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner  # noqa: E402
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR  # noqa: E402
from molmo_spaces.tasks.pact_place import (  # noqa: E402
    PactPlaceCorridorV1010FourObjectSampler,
)
from molmo_spaces.tasks.pact_place_contact_audit import PactPlaceContactAudit  # noqa: E402

try:
    import wandb  # type: ignore
except ImportError:
    wandb = None

_METRICS_JSONL: _Path | None = None
_PENDING_ROW_META: dict | None = None


class ACTPactPickNPlaceEvalConfig(FrankaSkinPactPlaceV1010FourObjectConfig):
    """V10.10 four-object place (OOD vs v1011d train); ACT/PACT swapped in. Exo+wrist."""

    policy_config: ACTPolicyConfig = ACTPolicyConfig()
    viz_sensor_rgb: bool = False
    filter_for_successful_trajectories: bool = False
    use_wandb: bool = False
    num_workers: int = 1
    save_videos: bool = False
    use_passive_viewer: bool = False
    output_dir: _Path = ASSETS_DIR / "datagen" / "act_pact_pick_n_place_eval"

    @property
    def tag(self) -> str:
        return "act_pact_pick_n_place_eval"


class _PlaceEvalRunner(ParallelRolloutRunner):
    """Attach the place contact audit so ACT rollouts still score hazard-bar hits."""

    @staticmethod
    def run_single_rollout(episode_seed, task, policy, **kwargs):
        audit = PactPlaceContactAudit()
        task._contact_audit_hook = audit
        rollout_started = time.perf_counter()
        success = ParallelRolloutRunner.run_single_rollout(
            episode_seed=episode_seed, task=task, policy=policy, **kwargs
        )
        try:
            _record_place_metric(
                task, bool(success), audit.summary(),
                rollout_wall_seconds=time.perf_counter() - rollout_started,
            )
        except Exception as e:
            print(f"[act-eval-pact-v2] contact metric capture failed: {e}")
        return success


def _record_place_metric(
    task, success: bool, audit: dict, *, rollout_wall_seconds: float | None = None,
) -> None:
    frames = audit.get("frames_with_contact") or {}
    totals = audit.get("contact_class_totals") or {}
    bar_frames = int(frames.get("hazard_bar") or 0)
    other_frames = int(frames.get("other_environment") or 0)
    clutter_frames = int(frames.get("clutter") or 0)
    suite = getattr(task, "sensor_suite", None) or getattr(task, "_sensor_suite", None)
    n_fresh = int(getattr(suite, "_fast_eval_n_fresh", 0) or 0)
    n_skip = int(getattr(suite, "_fast_eval_n_skip", 0) or 0)
    policy = getattr(task, "_registered_policy", None)
    rec = {
        "episode_idx": len(_EPISODE_METRICS),
        "success": int(success),
        "hit_bar": int(bar_frames > 0),
        "bar_contact_frames": bar_frames,
        "other_environment_frames": other_frames,
        "clutter_frames": clutter_frames,
        "collision_free": int(bool(audit.get("collision_free"))),
        "collision_free_task_success": int(
            bool(success) and bool(audit.get("collision_free"))
        ),
        "gripper_close_commanded": int(
            bool(getattr(policy, "gripper_close_commanded", False))
        ),
        "intrusion_side": str(
            (getattr(task, "scene_params", {}) or {}).get("pact_intrusion_side") or ""
        ),
        "contact_class_totals": totals,
        "first_contact_step": audit.get("first_contact_step") or {},
        "sensor_fresh_renders": n_fresh,
        "sensor_skipped_renders": n_skip,
        "fresh_observation_seconds": float(getattr(suite, "_fast_eval_skin_s", 0.0) or 0.0),
        "rollout_wall_seconds": rollout_wall_seconds,
    }
    if _PENDING_ROW_META:
        rec.update(_PENDING_ROW_META)
    _EPISODE_METRICS.append(rec)
    extra = f" renders={n_fresh} skip={n_skip}" if (n_fresh or n_skip) else ""
    print(
        f"[act-eval-pact-v2] ep{rec['episode_idx']:03d} success={success} "
        f"hit_bar={rec['hit_bar']} bar_frames={bar_frames} "
        f"other={other_frames} clutter={clutter_frames} "
        f"collision_free={rec['collision_free']} "
        f"grip_close={rec['gripper_close_commanded']} "
        f"side={rec['intrusion_side'] or '-'}{extra}",
        flush=True,
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
    strict = sum(int(m.get("collision_free_task_success", 0)) for m in _EPISODE_METRICS)
    if strict == 0:
        strict = sum(
            1 for m in _EPISODE_METRICS if int(m["success"]) and int(m["collision_free"])
        )
    grip = sum(int(m.get("gripper_close_commanded", 0)) for m in _EPISODE_METRICS)

    def _touched(class_name: str) -> int:
        return sum(
            1
            for m in _EPISODE_METRICS
            if int((m.get("contact_class_totals") or {}).get(class_name) or 0) > 0
        )

    grasp_n = _touched("grasp_target")
    tray_n = _touched("place_receptacle")
    clutter_n = _touched("clutter")
    return {
        "episodes": n,
        "success": successes,
        "success_rate": successes / n,
        "bar_hits": bar_hits,
        "bar_hit_rate": bar_hits / n,
        "collision_free": collision_free,
        "collision_free_rate": collision_free / n,
        "collision_rate": 1.0 - (collision_free / n),
        "collision_free_task_success": strict,
        "collision_free_task_success_rate": strict / n,
        "gripper_close_commanded": grip,
        "gripper_close_rate": grip / n,
        "grasp_target_episodes": grasp_n,
        "grasp_target_rate": grasp_n / n,
        "place_receptacle_episodes": tray_n,
        "place_receptacle_rate": tray_n / n,
        "clutter_episodes": clutter_n,
        "clutter_rate": clutter_n / n,
        "episodes_detail": list(_EPISODE_METRICS),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", "--checkpoint-dir", dest="ckpt_dir", required=True)
    p.add_argument("--ckpt_name", default="policy_best.ckpt")
    p.add_argument(
        "--output_dir",
        default="/home/jaydv/code/prox_learning/eval_output/pact_pick_n_place_v2",
    )
    p.add_argument("--num_rollouts", type=int, default=2)
    p.add_argument("--house_ind", type=int, default=1)
    p.add_argument(
        "--spread_cells",
        action="store_true",
        help=(
            "24 V10.10 cells (family × side × pose). "
            "n=48 → 2/cell; n=2 with this flag still runs 24 (1/cell)."
        ),
    )
    p.add_argument("--task_horizon", type=int, default=1050)
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
        "--fast_prox_rays",
        action="store_true",
        default=True,
        help="Default. Skin via mj_multiRay (not 40 EGL update_scene).",
    )
    p.add_argument(
        "--egl_prox",
        action="store_true",
        help="40-cam EGL rasterizer. Default is mj_multiRay.",
    )
    p.add_argument(
        "--save_trajectories",
        action="store_true",
        help="Keep datagen MP4/HDF5 (slow, OOM-prone). Default is metrics-only.",
    )
    p.add_argument("--live", "--render", "--viewer", dest="live", action="store_true")
    p.add_argument(
        "--molmospaces_root",
        type=str,
        default=str(DEFAULT_WORKTREE),
        help="Worktree of molmospaces origin/main (V1010 sampler + v10_7 XMLs).",
    )
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="act-pact-pick-n-place-eval")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--arm-name", dest="arm_name", default="")
    return p.parse_args()


def _bind_scenes(eval_cfg, args) -> tuple[list[str], str]:
    scenes_dir = resolve_v1010_scenes_dir(_MOLMO_ROOT)
    print(f"[act-eval-pact-v2] scenes_dir={scenes_dir}", flush=True)
    paths = [str(p) for p in v1010_scene_paths(scenes_dir)]
    eval_cfg.task_sampler_config.task_sampler_class = (
        PactPlaceCorridorV1010FourObjectSampler
    )
    eval_cfg.task_sampler_config.scene_xml_paths = paths
    if args.spread_cells:
        per, total = spread_episode_count(args.num_rollouts)
        eval_cfg.task_sampler_config.house_inds = list(range(N_V1010_CELLS))
        eval_cfg.task_sampler_config.samples_per_house = per
        protocol = "spread_24_cells"
        print(
            f"[act-eval-pact-v2] spread_cells requested_n={args.num_rollouts} "
            f"→ {N_V1010_CELLS} houses × {per} = {total} episodes",
            flush=True,
        )
        if total != args.num_rollouts:
            print(
                "[act-eval-pact-v2] WARNING: episode count is 24-cell multiple, "
                f"not {args.num_rollouts}. n=48 is 2/cell; n=2+spread is 24.",
                flush=True,
            )
    else:
        family, side, pose = v1010_cell(args.house_ind)
        eval_cfg.task_sampler_config.house_inds = [args.house_ind]
        eval_cfg.task_sampler_config.samples_per_house = args.num_rollouts
        protocol = f"pin_house_{args.house_ind}_{family}_{side}_{pose}"
        print(
            f"[act-eval-pact-v2] pin house_ind={args.house_ind} "
            f"cell={family}|{side}|{pose} n={args.num_rollouts} "
            "(one cell; v1011d train covers all 24 — use --spread_cells for that)",
            flush=True,
        )
    return paths, protocol


def main() -> None:
    args = parse_args()
    _EPISODE_METRICS.clear()
    if not args.temp_agg_off:
        print(
            "[act-eval-pact-v2] WARNING: --temp_agg_off not set. Open-loop chunking "
            "is the valid PACT path; temp-agg-on almost ignores live skin."
        )
    try:
        assert_v12_wraps_center(V12_XML)
        print(f"[act-eval-pact-v2] 7d1ea35 v12.xml wraps {V12_XML.name} → v10_7_center")
    except Exception as exc:
        print(f"[act-eval-pact-v2] WARNING: v12 xml check failed: {exc}")

    eval_cfg = ACTPactPickNPlaceEvalConfig()
    eval_cfg.task_horizon = args.task_horizon
    eval_cfg.end_on_success = False
    eval_cfg.output_dir = Path(args.output_dir).resolve()
    eval_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    eval_cfg.camera_config = FrankaSkinHybridCameraSystem()
    if hasattr(eval_cfg.robot_config, "action_noise_config"):
        noise = eval_cfg.robot_config.action_noise_config
        if noise is not None and hasattr(noise, "enabled"):
            noise.enabled = False

    paths, protocol = _bind_scenes(eval_cfg, args)

    live = args.live
    if live and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print("[act-eval-pact-v2] --live requested but no display; headless")
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
    pc.camera_names = ("exo_camera_1", "wrist_camera")

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
            f"[act-eval-pact-v2] PACT ckpt -> proximity ON "
            f"(feature={pc.prox_feature}, layout={pc.prox_layout}, "
            f"K={pc.prox_tokens_per_sensor}, pool={pc.prox_pool})"
        )
    elif args.use_proximity:
        pc.use_proximity = True
        pc.prox_feature = args.prox_feature
        pc.prox_layout = args.prox_layout
        pc.prox_tokens_per_sensor = args.prox_tokens_per_sensor
        pc.prox_encoder_ckpt = args.prox_encoder_ckpt
        print("[act-eval-pact-v2] proximity FORCED ON via CLI")
    else:
        print("[act-eval-pact-v2] vanilla ACT (no prox_config.json)")

    global _METRICS_JSONL
    _METRICS_JSONL = eval_cfg.output_dir / "episodes.jsonl"
    _configure_eval_cameras(eval_cfg, need_skin=bool(pc.use_proximity))
    if args.temp_agg_off:
        _install_chunk_gated_sensors()
    use_rays = bool(pc.use_proximity) and not bool(args.egl_prox)
    if use_rays:
        _install_raycast_proximity()
    elif pc.use_proximity:
        print(
            "[act-eval-pact-v2] WARNING: --egl_prox ON. Default is mj_multiRay.",
            flush=True,
        )
    if not args.save_trajectories:
        _install_metrics_only_hooks()
        print("[act-eval-pact-v2] metrics-only eval (no MP4/HDF5, policy loaded once)")

    eval_cfg.save_config()
    xml0 = paths[args.house_ind % len(paths)]
    print(
        f"[act-eval-pact-v2] molmospaces={_MOLMO_ROOT} "
        f"sampler={eval_cfg.task_sampler_config.task_sampler_class.__name__} "
        f"xml0={Path(xml0).name} cameras={pc.camera_names} "
        f"horizon={eval_cfg.task_horizon} protocol={protocol}"
    )
    print(f"[act-eval-pact-v2] writing {eval_cfg.output_dir}")
    print(
        "[act-eval-pact-v2] OOD: train sampler is "
        "PactPlaceCorridorV1011DRandomizedLayoutSampler "
        "(pact_place_corridor_v10_11d_randomized_clutter, 6 clutter + primitives). "
        "This process samples PactPlaceCorridorV1010FourObjectSampler "
        "(4 frozen household objects, no slots 08/09). "
        "origin/main has no V1011D class. Fair eval needs worktree 70dedc0. "
        "v12 kitchen overlay is OFF. Do not cite this run as in-distribution."
    )

    os.environ.pop("WANDB_RUN_NAME", None)
    os.environ.pop("WANDB_PROJECT_NAME", None)

    if args.use_wandb:
        if wandb is None:
            raise RuntimeError("wandb not installed")
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"eval_pact_pick_n_place_{int(time.time())}",
            config={"ckpt_dir": pc.ckpt_dir, "num_rollouts": args.num_rollouts},
        )

    policy = ACTInferencePolicy(eval_cfg)
    policy.prepare_model()
    runner = _PlaceEvalRunner(eval_cfg)
    evaluation_started = time.perf_counter()
    success, total = runner.run(preloaded_policy=policy)
    evaluation_wall_seconds = time.perf_counter() - evaluation_started
    rate = (success / total) if total else 0.0
    print(f"[act-eval-pact-v2] success {success}/{total}  ({rate*100:.1f}%)")
    collision = _summarize_place_metrics()
    if collision is not None:
        print(
            f"[act-eval-pact-v2] bar_hit {collision['bar_hits']}/{collision['episodes']} "
            f"({collision['bar_hit_rate']*100:.1f}%)  "
            f"collision_free {collision['collision_free']}/{collision['episodes']} "
            f"({collision['collision_free_rate']*100:.1f}%)  "
            f"strict {collision.get('collision_free_task_success', 0)}/"
            f"{collision['episodes']}  "
            f"grip_close {collision.get('gripper_close_commanded', 0)}/"
            f"{collision['episodes']}  "
            f"grasp {collision.get('grasp_target_episodes', 0)}/"
            f"{collision['episodes']}  "
            f"tray {collision.get('place_receptacle_episodes', 0)}/"
            f"{collision['episodes']}"
        )

    summary = {
        "ckpt_dir": pc.ckpt_dir,
        "ckpt_name": pc.ckpt_name,
        "task": "pact_pick_n_place_v2",
        "train_env": "pact_place_corridor_v10_11d_randomized_clutter",
        "train_sampler_class": "PactPlaceCorridorV1011DRandomizedLayoutSampler",
        "eval_env": "pact_place_corridor_v10_10_four_object",
        "eval_is_ood": True,
        "eval_xml_family": "pact_place_corridor_v10_7",
        "v12_wrapper": str(V12_XML),
        "v12_commit": "7d1ea352a3d204eac1bd3661d01ab628f7b7c188",
        "molmospaces_root": str(_MOLMO_ROOT),
        "molmospaces_pin": MOLMOSPACES_V1010_SHA,
        "task_sampler_class": eval_cfg.task_sampler_config.task_sampler_class.__name__,
        "camera_names": list(pc.camera_names),
        "num_rollouts_requested": args.num_rollouts,
        "task_horizon": args.task_horizon,
        "chunk_size": pc.chunk_size,
        "temp_agg_off": pc.temp_agg_off,
        "use_proximity": pc.use_proximity,
        "proximity_backend": "rays" if use_rays else ("egl" if pc.use_proximity else "none"),
        "metrics_only": not args.save_trajectories,
        "export_sensor_filter": not args.save_trajectories,
        "evaluation_wall_seconds": evaluation_wall_seconds,
        "protocol": protocol,
        "spread_cells": bool(args.spread_cells),
        "house_inds": list(eval_cfg.task_sampler_config.house_inds),
        "samples_per_house": eval_cfg.task_sampler_config.samples_per_house,
        "kitchen_overlay": False,
        "arm_name": args.arm_name or None,
        "success": int(success),
        "total": int(total),
        "success_rate": float(rate),
        "collision": collision,
    }
    out = eval_cfg.output_dir / "eval_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"[act-eval-pact-v2] wrote {out}")


if __name__ == "__main__":
    main()
