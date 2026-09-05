"""Chunk-gate / raycast / metrics-only hooks for in-env place eval.

Lazy-imports molmospaces inside the install functions so hallway v5 (977acd6)
and v1011d (origin/main) can share this file without pinning a worktree here.
"""
from __future__ import annotations

import sys
import time

_RENDER_CLASS_NAMES = frozenset(
    {
        "CameraParameterSensor",
        "CameraSensor",
        "DepthSensor",
        "ProximityDepthBufferSensor",
        "ProximityVizDepthSensor",
        "ProximityVizRGBSensor",
    }
)
_RGB_DEPTH_OFF = frozenset({"wrist_camera", "exo_camera_1", "table_camera"})
_EXPORT_SENSOR_CLASS_NAMES = frozenset({"ObjectImagePointsSensor", "EnvStateSensor"})


def _without_export_sensors(sensors):
    """Remove dataset annotations unused by ACT inputs or the task's judges.

    ObjectImagePointsSensor performs segmentation rendering through every configured
    camera on every step. It bypasses the RGB/proximity chunk gate. EnvStateSensor
    walks all scene bodies to build an export that metrics-only evaluation discards.
    Keep all other sensors, including stateful grasp sensors and policy observations.
    """
    return [s for s in sensors if type(s).__name__ not in _EXPORT_SENSOR_CLASS_NAMES]


def _install_metrics_only_sensor_filter() -> None:
    import molmo_spaces.env.sensors as sensors_module

    original = sensors_module.get_core_sensors
    if getattr(original, "_pact_metrics_only", False):
        return

    def get_core_sensors(*args, **kwargs):
        sensors = original(*args, **kwargs)
        kept = _without_export_sensors(sensors)
        removed = [s.uuid for s in sensors if type(s).__name__ in _EXPORT_SENSOR_CLASS_NAMES]
        if removed:
            print(f"[act-eval-place] metrics-only: omit export sensors {removed}", flush=True)
        return kept

    get_core_sensors._pact_metrics_only = True
    sensors_module.get_core_sensors = get_core_sensors


def _policy_wants_fresh(task) -> bool:
    policy = getattr(task, "_registered_policy", None)
    if policy is not None and hasattr(policy, "needs_fresh_policy_observation"):
        return bool(policy.needs_fresh_policy_observation())
    return True


def _is_render_sensor(sensor, render_types: tuple) -> bool:
    if isinstance(sensor, render_types):
        return True
    return type(sensor).__name__ in _RENDER_CLASS_NAMES


def _is_prox_depth_sensor(sensor, prox_types: tuple) -> bool:
    if prox_types and isinstance(sensor, prox_types):
        return True
    return type(sensor).__name__ == "ProximityDepthBufferSensor"


def _heartbeat(suite, extra: str = "") -> None:
    n_fresh = int(getattr(suite, "_fast_eval_n_fresh", 0) or 0)
    n_skip = int(getattr(suite, "_fast_eval_n_skip", 0) or 0)
    total = n_fresh + n_skip
    if total in (1, 10, 50) or (total > 0 and total % 100 == 0):
        skin_s = float(getattr(suite, "_fast_eval_skin_s", 0.0) or 0.0)
        print(
            f"[act-eval-place] sensors fresh={n_fresh} skip={n_skip} "
            f"skin_s={skin_s:.2f}{extra}",
            flush=True,
        )


def _patch_all_named_get_observation(class_name: str) -> int:
    """Patch ``class_name`` only on ``molmo_spaces.*`` modules.

    Walking every ``sys.modules`` entry ``getattr``s HuggingFace lazy
    image processors and prints a wall of ``ProximityDepthBufferSensor``
    alias warnings. Those classes are not ours.
    """
    patched = 0
    seen: set[int] = set()
    for mod_name, mod in list(sys.modules.items()):
        if not isinstance(mod_name, str) or not mod_name.startswith("molmo_spaces"):
            continue
        cls = getattr(mod, class_name, None)
        if not isinstance(cls, type):
            continue
        key = id(cls)
        if key in seen or not hasattr(cls, "get_observation"):
            continue
        seen.add(key)
        orig = cls.get_observation

        def get_observation(self, env, task, *args, _orig=orig, **kwargs):
            if (not _policy_wants_fresh(task)) and getattr(
                self, "_fast_eval_last_frame", None
            ) is not None:
                return self._fast_eval_last_frame
            out = _orig(self, env, task, *args, **kwargs)
            self._fast_eval_last_frame = out
            return out

        cls.get_observation = get_observation
        patched += 1
    return patched


def _install_chunk_gated_sensors() -> None:
    """Skip RGB / 8x8 skin EGL on steps the open-loop chunk will ignore.

    ``SensorSuite.get_observations`` is the 0.75 s/step PACT cost. Wrist + 40
    proximity cameras still render on every query (chunk boundary). Physics and
    the 2 ms contact audit are unchanged.

    Also patches ``ProximityDepthBufferSensor.get_observation`` by class name so
    a second ``molmo_spaces`` on ``sys.path`` cannot bypass ``isinstance``.
    Fresh queries batch all 40 skins into one ``record_proximity_depths`` call.
    """
    from molmo_spaces.env.abstract_sensors import SensorSuite
    from molmo_spaces.env.sensors_cameras import (
        CameraParameterSensor,
        CameraSensor,
        DepthSensor,
        ProximityDepthBufferSensor,
        ProximityVizDepthSensor,
        ProximityVizRGBSensor,
    )

    render_types = (
        CameraParameterSensor,
        CameraSensor,
        DepthSensor,
        ProximityDepthBufferSensor,
        ProximityVizDepthSensor,
        ProximityVizRGBSensor,
    )
    prox_types = (ProximityDepthBufferSensor,)
    orig = SensorSuite.get_observations
    n_prox_patch = _patch_all_named_get_observation("ProximityDepthBufferSensor")

    def get_observations(self, env, task, **kwargs):
        fresh = _policy_wants_fresh(task)
        last = getattr(self, "_fast_eval_last_obs", None)
        if (not fresh) and last is not None:
            obs = dict(last)
            for uuid, sensor in self.sensors.items():
                if _is_render_sensor(sensor, render_types):
                    continue
                obs[uuid] = sensor.get_observation(env=env, task=task, **kwargs)
            self._fast_eval_n_skip = getattr(self, "_fast_eval_n_skip", 0) + 1
            _heartbeat(self)
            return obs
        prox_names = [
            sensor.camera_name
            for sensor in self.sensors.values()
            if _is_prox_depth_sensor(sensor, prox_types)
        ]
        t0 = time.perf_counter()
        if prox_names and hasattr(env, "reset_proximity_depth_buffer"):
            env.reset_proximity_depth_buffer(prox_names)
            if hasattr(env, "record_proximity_depths"):
                env.record_proximity_depths(prox_names)
        obs = orig(self, env, task, **kwargs)
        dt = time.perf_counter() - t0
        self._fast_eval_last_obs = obs
        self._fast_eval_n_fresh = getattr(self, "_fast_eval_n_fresh", 0) + 1
        self._fast_eval_skin_s = getattr(self, "_fast_eval_skin_s", 0.0) + dt
        n_fresh = self._fast_eval_n_fresh
        if prox_names and (n_fresh <= 3 or n_fresh % 10 == 0):
            print(
                f"[act-eval-place] skin query #{n_fresh} n_cam={len(prox_names)} "
                f"{dt:.3f}s",
                flush=True,
            )
        _heartbeat(self)
        return obs

    SensorSuite.get_observations = get_observations
    print(
        "[act-eval-place] chunk-gated sensors ON "
        f"(RGB/skin only on chunk query; prox_get_obs patches={n_prox_patch})",
        flush=True,
    )


def _pixel_dirs_cam(h: int = 8, w: int = 8, fovy_deg: float = 45.0):
    import numpy as np

    yfov = float(np.tan(np.radians(fovy_deg) / 2.0))
    ys, xs = np.meshgrid(
        1.0 - 2.0 * (np.arange(h) + 0.5) / h,
        2.0 * (np.arange(w) + 0.5) / w - 1.0,
        indexing="ij",
    )
    dirs = np.stack([xs * yfov, ys * yfov, -np.ones_like(xs)], axis=-1)
    dirs = dirs.reshape(-1, 3)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    return dirs.astype(np.float64)


def _install_raycast_proximity() -> None:
    """Replace 40 EGL ``update_scene`` calls with ``mj_multiRay``.

    Same 45° 8×8 grid, geom group 2 hidden. Not bit-identical to the EGL
    rasterizer (pixel is a fat cone; a ray is the pixel center). Use for
    iteration. Paper table stays on EGL.
    """
    import numpy as np
    import mujoco
    from molmo_spaces.env.env import CPUMujocoEnv

    orig = CPUMujocoEnv.record_proximity_depths
    dirs_by_fovy: dict[float, object] = {}
    geomgroup = np.ones((6, 1), dtype=np.uint8)
    geomgroup[2, 0] = 0
    cutoff = 10.0
    _multi_wants_normal = False
    if hasattr(mujoco, "mj_multiRay"):
        import inspect

        try:
            _multi_wants_normal = "normal" in inspect.signature(
                mujoco.mj_multiRay
            ).parameters
        except (TypeError, ValueError):
            _multi_wants_normal = True

    def record_proximity_depths(self, camera_names):
        model = self.mj_model
        data = self.current_data
        for name in camera_names:
            full = self._proximity_cam_full_name(name) if hasattr(
                self, "_proximity_cam_full_name"
            ) else None
            if not full:
                orig(self, [name])
                continue
            cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, full)
            if cid < 0:
                orig(self, [name])
                continue
            fovy = float(model.cam_fovy[cid])
            dirs_cam = dirs_by_fovy.get(fovy)
            if dirs_cam is None:
                dirs_cam = _pixel_dirs_cam(fovy_deg=fovy)
                dirs_by_fovy[fovy] = dirs_cam
            nray = int(dirs_cam.shape[0])
            pos = np.asarray(data.cam_xpos[cid], dtype=np.float64)
            rot = np.asarray(data.cam_xmat[cid], dtype=np.float64).reshape(3, 3)
            dirs_w = dirs_cam @ rot.T
            look = -rot[:, 2]
            pnt = np.ascontiguousarray(
                (pos + 1e-4 * look).reshape(3, 1), dtype=np.float64
            )
            geomid = np.zeros((nray, 1), dtype=np.int32)
            dist = np.empty((nray, 1), dtype=np.float64)
            vec = np.ascontiguousarray(dirs_w.reshape(-1, 1), dtype=np.float64)
            if hasattr(mujoco, "mj_multiRay"):
                if _multi_wants_normal:
                    mujoco.mj_multiRay(
                        model,
                        data,
                        pnt,
                        vec,
                        geomgroup,
                        1,
                        -1,
                        geomid,
                        dist,
                        None,
                        nray,
                        cutoff,
                    )
                else:
                    mujoco.mj_multiRay(
                        model,
                        data,
                        pnt,
                        vec,
                        geomgroup,
                        1,
                        -1,
                        geomid,
                        dist,
                        nray,
                        cutoff,
                    )
            else:
                one_gid = np.zeros((1, 1), dtype=np.int32)
                for i in range(nray):
                    dist[i, 0] = mujoco.mj_ray(
                        model,
                        data,
                        pnt,
                        np.ascontiguousarray(dirs_w[i].reshape(3, 1), dtype=np.float64),
                        geomgroup,
                        1,
                        -1,
                        one_gid,
                    )
            depth = dist.reshape(8, 8).astype(np.float32)
            depth[depth < 0] = np.float32(cutoff)
            self._proximity_depth_frames.setdefault(name, []).append(depth)

    CPUMujocoEnv.record_proximity_depths = record_proximity_depths
    print(
        "[act-eval-place] proximity mj_multiRay ON "
        "(iteration only; not the paper-table EGL rasterizer)",
        flush=True,
    )


def _install_metrics_only_hooks() -> None:
    """Do not keep RGB/depth histories or write MP4/HDF5.

    The datagen pipeline otherwise retains every episode until the house
    finishes, which OOM-kills a 50-ep eval on a 62 GB box.
    """
    import molmo_spaces.data_generation.pipeline as pipeline
    from molmo_spaces.tasks.task import BaseMujocoTask

    _install_metrics_only_sensor_filter()

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
    """RGB without depth. PACT: skin at policy rate, not 60 Hz substeps.

    Hallway v5 is wrist-only. v1011d also has ``exo_camera_1``. Depth is unused
    by ACT either way.
    """
    eval_cfg.proximity_sensor_period_ms = 0.0
    cams = []
    for cam in list(eval_cfg.camera_config.cameras):
        if (not need_skin) and getattr(cam, "is_proximity_sensor", False):
            continue
        if getattr(cam, "name", None) in _RGB_DEPTH_OFF:
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


def _install_contract_sensor_gate() -> None:
    """Gate RGB by action queries; retain consecutive skin for readout policies.

    Unlike the legacy gate, preserve the native 60-Hz depth buffer on query
    steps. History encoders keep the native buffer every control step. Never enable raycast
    replacement or reset a populated substep buffer before it is consumed.
    """
    from molmo_spaces.env.abstract_sensors import SensorSuite
    from molmo_spaces.tasks.task import BaseMujocoTask

    original_observations = SensorSuite.get_observations
    original_step = BaseMujocoTask.step
    if getattr(original_step, '_pact_contract_gate', False):
        return

    def wants(task, modality):
        policy = getattr(task, '_registered_policy', None)
        method = getattr(policy, f'needs_fresh_{modality}_observation', None)
        return bool(method()) if method is not None else _policy_wants_fresh(task)

    def get_observations(self, env, task, **kwargs):
        last = getattr(self, '_contract_last_observation', None)
        fresh = wants(task, 'camera')
        if fresh or last is None:
            observations = original_observations(self, env, task, **kwargs)
            self._contract_last_observation = observations
            return observations
        observations = dict(last)
        for uuid, sensor in self.sensors.items():
            if (type(sensor).__name__ not in _RENDER_CLASS_NAMES or
                    (_is_prox_depth_sensor(sensor, ()) and wants(task, 'proximity'))):
                observations[uuid] = sensor.get_observation(env=env, task=task, **kwargs)
        self._contract_last_observation = observations
        return observations

    def step(self, *args, **kwargs):
        cameras = self._proximity_camera_names
        # policy.get_action has already advanced its step index. This decides
        # whether the observation returned by THIS step will be consumed next.
        if not wants(self, 'proximity'):
            self._proximity_camera_names = []
        try:
            return original_step(self, *args, **kwargs)
        finally:
            self._proximity_camera_names = cameras

    step._pact_contract_gate = True
    BaseMujocoTask.step = step
    SensorSuite.get_observations = get_observations
