"""Eval-time per-link Poisson sensor keep for the hybrid 40-SPAD skin.

Used to test whether PACT is sensor-agnostic: keep a random subset of SPADs
per link, leave the 40-token layout intact, fill dropped depths with D_MAX
(0.5 m = no return). Never fill 0 (that reads as contact).

Keep fraction p:
  p >= 1  keep all (exact; Poisson(n) would still drop sensors)
  p <= 0  drop all (required control)
  else    per link L with n_L sensors:
            k = clip(Poisson(p * n_L), 0, n_L)
            then uniform sample k indices without replacement

Clip pulls E[k] below lambda when lambda is near n_L. Small links are noisy
(link5_front n=4, p=0.5, lambda=2: P(k=0)=13.5%). Log realized per-link
counts and report mean +/- sd across episodes. Do not claim realized keep
equals nominal p.

Groups use the prefix before ``_sensor_``, so link5_back (6) and
link5_front (4) are two links. Order is HYBRID_SKIN_SENSOR_ORDER.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from hybrid_skin_sensors import D_MAX, DEAD_PIXEL_M, HYBRID_SKIN_SENSOR_ORDER

MASK_SEED_STRIDE = 1_000_003
_GRAY_DROPPED = (80, 80, 80)
_MOSAIC_BG = 18


def link_prefix(name: str) -> str:
    if "_sensor_" in name:
        return name.rsplit("_sensor_", 1)[0]
    return name


def group_indices(
    order: Sequence[str] | None = None,
) -> OrderedDict[str, list[int]]:
    names = list(order if order is not None else HYBRID_SKIN_SENSOR_ORDER)
    groups: OrderedDict[str, list[int]] = OrderedDict()
    for i, name in enumerate(names):
        groups.setdefault(link_prefix(name), []).append(i)
    return groups


def mask_rng(
    mask_seed: int,
    episode_idx: int,
    *,
    fixed: bool = False,
) -> np.random.Generator:
    seed = int(mask_seed)
    if not fixed:
        seed = seed + MASK_SEED_STRIDE * int(episode_idx)
    return np.random.default_rng(seed)


def poisson_keep_mask(
    order: Sequence[str],
    p: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """bool[len(order)] True = keep. See module docstring for the p rule."""
    n = len(order)
    mask = np.zeros(n, dtype=bool)
    frac = float(p)
    if frac >= 1.0:
        mask[:] = True
        return mask
    if frac <= 0.0:
        return mask
    groups = group_indices(order)
    for _link, idxs in groups.items():
        n_l = len(idxs)
        lam = frac * n_l
        k = int(rng.poisson(lam))
        k = max(0, min(k, n_l))
        if k == 0:
            continue
        chosen = rng.choice(n_l, size=k, replace=False)
        for j in chosen:
            mask[idxs[int(j)]] = True
    return mask


def apply_mask(
    prox: np.ndarray,
    mask: np.ndarray,
    fill: float = D_MAX,
) -> np.ndarray:
    """Copy prox and write fill on dropped sensors.

    ``prox`` is (S, 8, 8) or (..., S, 8, 8). ``mask`` is bool[S].
    """
    fill_v = float(fill)
    if fill_v <= 0.0:
        raise ValueError(
            "sensor keep fill must be > 0 (0 reads as contact). Use D_MAX=0.5 m."
        )
    if fill_v < DEAD_PIXEL_M:
        raise ValueError(
            f"sensor keep fill {fill_v} m is below DEAD_PIXEL_M={DEAD_PIXEL_M} m "
            "and would look like a dead/contact pixel. Use D_MAX=0.5 m."
        )
    out = np.array(prox, dtype=np.float32, copy=True)
    keep = np.asarray(mask, dtype=bool)
    if out.ndim < 3:
        raise ValueError(f"prox must be (..., S, 8, 8); got {out.shape}")
    n_s = out.shape[-3]
    if keep.shape != (n_s,):
        raise ValueError(f"mask shape {keep.shape} != ({n_s},) for prox {out.shape}")
    dropped = ~keep
    if not np.any(dropped):
        return out
    # Broadcast over leading batch / history axes: (..., S, 8, 8)
    out[..., dropped, :, :] = np.float32(fill_v)
    return out


def mask_record(
    order: Sequence[str],
    mask: np.ndarray,
    *,
    keep_frac: float,
    mask_seed: int,
    mask_fixed: bool,
    episode_idx: int,
) -> dict:
    names = list(order)
    keep = np.asarray(mask, dtype=bool)
    if keep.shape != (len(names),):
        raise ValueError(f"mask shape {keep.shape} != ({len(names)},)")
    groups = group_indices(names)
    per_link = {link: int(keep[idxs].sum()) for link, idxs in groups.items()}
    per_link_n = {link: len(idxs) for link, idxs in groups.items()}
    n_keep = int(keep.sum())
    n = len(names)
    return {
        "sensor_keep_frac": float(keep_frac),
        "sensor_mask_seed": int(mask_seed),
        "sensor_mask_fixed": bool(mask_fixed),
        "sensor_keep_episode_idx": int(episode_idx),
        "sensor_keep_names": [names[i] for i, flag in enumerate(keep) if flag],
        "sensor_keep_count": n_keep,
        "sensor_keep_n": n,
        "sensor_keep_realized_frac": (n_keep / n) if n else 0.0,
        "sensor_keep_per_link": per_link,
        "sensor_keep_per_link_n": per_link_n,
    }


def _sd(vals: np.ndarray) -> float:
    if vals.size <= 1:
        return 0.0
    return float(vals.std(ddof=1))


def summarize_keep_records(records: Iterable[Mapping]) -> dict | None:
    recs = [dict(r) for r in records if "sensor_keep_count" in r]
    if not recs:
        return None
    overall = np.array(
        [float(r["sensor_keep_realized_frac"]) for r in recs], dtype=np.float64
    )
    counts = np.array([int(r["sensor_keep_count"]) for r in recs], dtype=np.float64)
    n_ref = recs[0].get("sensor_keep_per_link_n") or {}
    per_link: dict[str, dict] = {}
    for link, n_l in n_ref.items():
        n_l = int(n_l)
        vals = []
        for rec in recs:
            kept = (rec.get("sensor_keep_per_link") or {}).get(link)
            if kept is None:
                continue
            vals.append(float(kept) / n_l if n_l else 0.0)
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        per_link[link] = {
            "mean": float(arr.mean()),
            "sd": _sd(arr),
            "n": n_l,
        }
    return {
        "sensor_keep_frac": recs[0].get("sensor_keep_frac"),
        "sensor_mask_fixed": bool(recs[0].get("sensor_mask_fixed", False)),
        "sensor_mask_seed": recs[0].get("sensor_mask_seed"),
        "n_episodes": len(recs),
        "realized_frac_mean": float(overall.mean()),
        "realized_frac_sd": _sd(overall),
        "realized_count_mean": float(counts.mean()),
        "realized_count_sd": _sd(counts),
        "per_link": per_link,
    }


def _as_bgr_uint8(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img)
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    if x.ndim != 3:
        raise ValueError(f"RGB image must be HxWxC; got {x.shape}")
    if x.shape[-1] == 4:
        x = x[..., :3]
    if x.shape[-1] != 3:
        raise ValueError(f"RGB image last dim must be 3; got {x.shape}")
    if x.dtype != np.uint8:
        xf = x.astype(np.float32)
        if float(np.nanmax(xf)) <= 1.0:
            x = (np.clip(xf, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            x = np.clip(xf, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(x[..., ::-1])  # RGB -> BGR


def _depth_bgr(
    depth: np.ndarray, *, near: float = 0.05, far: float = 2.0
) -> np.ndarray:
    import cv2

    d = np.asarray(depth, dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    if d.ndim != 2:
        raise ValueError(f"depth must be HxW; got {d.shape}")
    norm = np.clip((d - near) / max(far - near, 1e-6), 0.0, 1.0)
    u8 = (norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)


def render_sensor_mosaic(
    d8s: np.ndarray,
    names: Sequence[str],
    mask: np.ndarray | None,
    *,
    out_w: int,
    out_h: int,
    near: float = 0.05,
    far: float = D_MAX,
) -> np.ndarray:
    """(S, 8, 8) metres -> BGR mosaic, one row per link. Dropped tiles gray."""
    import cv2

    canvas = np.full((out_h, out_w, 3), _MOSAIC_BG, np.uint8)
    if d8s is None or len(d8s) == 0:
        cv2.putText(
            canvas,
            "no proximity",
            (16, out_h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (140, 140, 140),
            1,
            cv2.LINE_AA,
        )
        return canvas
    keep = (
        np.ones(len(names), dtype=bool)
        if mask is None
        else np.asarray(mask, dtype=bool)
    )
    groups = group_indices(names)
    n_rows = max(len(groups), 1)
    pad = 3
    caption_h = 12
    link_w = 70
    grid_w = max(out_w - link_w, 32)
    for r, (link, idxs) in enumerate(groups.items()):
        y0 = r * out_h // n_rows
        y1 = (r + 1) * out_h // n_rows
        row_h = y1 - y0
        n_c = max(len(idxs), 1)
        cell_w = max(8, (grid_w - pad * (n_c + 1)) // n_c)
        cell_h = max(8, row_h - caption_h - pad)
        cv2.putText(
            canvas,
            link,
            (out_w - link_w + 2, y0 + min(14, row_h - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (120, 200, 255),
            1,
            cv2.LINE_AA,
        )
        for c, i in enumerate(idxs):
            x = pad + c * (cell_w + pad)
            y = y0 + caption_h
            y2, x2 = min(y + cell_h, out_h), min(x + cell_w, grid_w)
            if x2 <= x or y2 <= y:
                continue
            if i >= len(keep) or not bool(keep[i]):
                tile = np.full((cell_h, cell_w, 3), _GRAY_DROPPED, np.uint8)
            else:
                patch = np.asarray(d8s[i], np.float32)
                valid = np.isfinite(patch) & (patch >= DEAD_PIXEL_M)
                norm = np.clip((patch - near) / max(far - near, 1e-6), 0.0, 1.0)
                u8 = (norm * 255.0).astype(np.uint8)
                col = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
                col[~valid] = 40
                tile = cv2.resize(
                    col, (cell_w, cell_h), interpolation=cv2.INTER_NEAREST
                )
            canvas[y:y2, x:x2] = tile[: y2 - y, : x2 - x]
            label = str(names[i]).replace("sensor_", "s")
            cv2.putText(
                canvas,
                label,
                (x, y0 + caption_h - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.28,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )
    return canvas


def compose_first_frame_rgbd_sensors(
    *,
    rgb_by_cam: Mapping[str, np.ndarray],
    depth_by_cam: Mapping[str, np.ndarray],
    cameras: Sequence[str],
    prox: np.ndarray | None,
    names: Sequence[str],
    mask: np.ndarray | None,
    caption: str = "",
    panel_h: int = 240,
    mosaic_h: int = 280,
    min_width: int = 640,
) -> np.ndarray:
    """BGR uint8: per-cam RGB|depth row, then 8x8 sensor mosaic (dropped gray)."""
    import cv2

    cams = list(cameras)
    if not cams:
        raise ValueError("compose_first_frame_rgbd_sensors needs cameras")
    rows = []
    target_h = int(panel_h)
    for cam in cams:
        if cam not in rgb_by_cam:
            raise KeyError(f"missing RGB for {cam!r}")
        if cam not in depth_by_cam:
            raise KeyError(f"missing depth for {cam!r}")
        rgb = _as_bgr_uint8(rgb_by_cam[cam])
        depth = _depth_bgr(depth_by_cam[cam])
        h, w = rgb.shape[:2]
        scale = target_h / max(h, 1)
        new_w = max(1, int(round(w * scale)))
        rgb_r = cv2.resize(rgb, (new_w, target_h), interpolation=cv2.INTER_AREA)
        dh, dw = depth.shape[:2]
        dscale = target_h / max(dh, 1)
        depth_r = cv2.resize(
            depth,
            (max(1, int(round(dw * dscale))), target_h),
            interpolation=cv2.INTER_NEAREST,
        )
        gap = np.full((target_h, 8, 3), _MOSAIC_BG, np.uint8)
        label = np.full((22, rgb_r.shape[1] + 8 + depth_r.shape[1], 3), 12, np.uint8)
        cv2.putText(
            label,
            f"{cam}  RGB | depth",
            (8, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        pair = np.concatenate([rgb_r, gap, depth_r], axis=1)
        if pair.shape[1] < label.shape[1]:
            pad = np.full(
                (pair.shape[0], label.shape[1] - pair.shape[1], 3),
                _MOSAIC_BG,
                np.uint8,
            )
            pair = np.concatenate([pair, pad], axis=1)
        elif pair.shape[1] > label.shape[1]:
            extra = np.full(
                (label.shape[0], pair.shape[1] - label.shape[1], 3),
                12,
                np.uint8,
            )
            label = np.concatenate([label, extra], axis=1)
        rows.append(np.concatenate([label, pair], axis=0))

    width = max(max(r.shape[1] for r in rows), int(min_width))
    padded = []
    for row in rows:
        if row.shape[1] < width:
            pad = np.full(
                (row.shape[0], width - row.shape[1], 3), _MOSAIC_BG, np.uint8
            )
            row = np.concatenate([row, pad], axis=1)
        padded.append(row)
    top = np.concatenate(padded, axis=0)
    mosaic = render_sensor_mosaic(
        np.asarray(prox) if prox is not None else np.zeros((0, 8, 8), np.float32),
        list(names),
        mask,
        out_w=width,
        out_h=int(mosaic_h),
    )
    cap_h = 28 if caption else 0
    cap = np.full((cap_h, width, 3), 12, np.uint8)
    if caption:
        cv2.putText(
            cap,
            caption[:140],
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
    return np.concatenate([cap, top, mosaic], axis=0) if cap_h else np.concatenate(
        [top, mosaic], axis=0
    )


def write_first_frame_png(path: Path | str, image_bgr: np.ndarray) -> Path:
    import cv2

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out), image_bgr)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for {out}")
    return out
