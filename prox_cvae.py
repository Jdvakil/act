"""Frozen proximity-CVAE feature extractor for P+ACT (PACT).

This is the bridge between the pretrained Safety-CVAE (``assets/safety/cvae_v3``,
trained by ``scripts/train_safety_cvae.py``) and the ACT policy. The CVAE maps the
40-sensor hybrid-skin depth (closeness) to a 7-DoF joint-space retreat delta. We reuse
it FROZEN as a skin-only feature extractor whose output is injected into the ACT
transformer encoder as extra conditioning token(s) — the "steering" signal.

Why not the CVAE's own encoder? Its encoder is a CONDITIONAL posterior q(z | skin, dq)
that needs the target retreat dq as input, so it cannot run at policy-inference time.
The genuinely skin-only path is the DECODER trunk (run at z = 0, the prior mean). We
expose two feature taps, selected by ``feature``:

  * "trunk"  (default): the 256-d hidden activation of the decoder trunk
                        (after dec.0->SiLU->dec.2->SiLU). The richest skin-only
                        representation the CVAE learned. feat_dim = 256.
  * "delta":  the CVAE's actual 7-DoF joint retreat-delta output, in real joint units
                        (decoder at z = 0, times label_scale). The literal steering
                        vector. feat_dim = n_out (= 7).

The extractor is frozen (eval() + requires_grad_(False)); only the ACT-side projection
(``input_proj_proximity``) and the extra positional embeddings learn. It lives OUTSIDE
the ACT policy so the policy checkpoint stays small and the CVAE weights load
deterministically (identical features at train and eval time).

Sensor ordering is the single most error-prone detail: ``cvae_v3/meta.json["sensors"]``
is the AUTHORITATIVE order the CVAE was trained on (note: link5_BACK precedes
link5_FRONT, which is the OPPOSITE of the env's ``_HYBRID_SKIN_SENSOR_NAMES`` tuple).
Both the dataset converter and the live eval MUST stack the 40 sensors in
``encoder.sensor_order``. This module is the one place that order is read from.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Must match scripts/train_safety_cvae.py exactly (closeness normalization range + the
# dead/invalid-pixel threshold). The CVAE sees garbage if these drift.
D_MAX = 0.5         # closeness divisor (m): c = clip(1 - d / D_MAX, 0, 1)
DEAD_PIXEL_M = 0.005  # depth below this is a dead/invalid return -> closeness 0
DEFAULT_CKPT = "/home/jaydv/code/prox_learning/assets/safety/cvae_v3"


class SafetyCVAE(nn.Module):
    """Vendored copy of scripts/train_safety_cvae.py:SafetyCVAE (state_dict-compatible).

    enc / dec are nn.Sequential([Linear, SiLU, Linear, SiLU, Linear]); the Linear layers
    sit at indices 0, 2, 4, matching the cvae_v3 state_dict keys enc.{0,2,4}.* and
    dec.{0,2,4}.*.
    """

    def __init__(self, n_in: int, n_out: int = 7, z_dim: int = 8) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.enc = nn.Sequential(
            nn.Linear(n_in + n_out, 512), nn.SiLU(),
            nn.Linear(512, 256), nn.SiLU(),
            nn.Linear(256, 2 * z_dim),
        )
        self.dec = nn.Sequential(
            nn.Linear(n_in + z_dim, 512), nn.SiLU(),
            nn.Linear(512, 256), nn.SiLU(),
            nn.Linear(256, n_out),
        )

    def act(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic head: decode at z = 0 (the prior mean). (N, n_in) -> (N, n_out)."""
        z = torch.zeros(x.shape[0], self.z_dim, device=x.device, dtype=x.dtype)
        return self.dec(torch.cat([x, z], -1))

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        """Decoder trunk hidden activation at z = 0: (N, n_in) -> (N, 256).

        Runs dec[0] (Linear 2568->512) -> dec[1] (SiLU) -> dec[2] (Linear 512->256)
        -> dec[3] (SiLU). This is the skin-only feature used to condition ACT.
        """
        z = torch.zeros(x.shape[0], self.z_dim, device=x.device, dtype=x.dtype)
        h = torch.cat([x, z], -1)
        h = self.dec[1](self.dec[0](h))   # Linear -> SiLU  (N, 512)
        h = self.dec[3](self.dec[2](h))   # Linear -> SiLU  (N, 256)
        return h


def featurize_np(prox: np.ndarray) -> np.ndarray:
    """(N, S, 8, 8) depths in meters -> (N, S*64) closeness in [0, 1].

    Identical to scripts/train_safety_cvae.py:featurize so the frozen CVAE sees the same
    distribution it was trained on.
    """
    d = prox.astype(np.float32)
    c = np.clip(1.0 - d / D_MAX, 0.0, 1.0)
    c[d < DEAD_PIXEL_M] = 0.0
    return c.reshape(len(c), -1)


def featurize_torch(prox: torch.Tensor) -> torch.Tensor:
    """Torch version of featurize. prox: (B, S, 8, 8) meters -> (B, S*64) closeness."""
    d = prox.float()
    c = torch.clamp(1.0 - d / D_MAX, 0.0, 1.0)
    c = torch.where(d < DEAD_PIXEL_M, torch.zeros_like(c), c)
    return c.reshape(c.shape[0], -1)


def feat_dim_for(ckpt_dir: str | Path = DEFAULT_CKPT, feature: str = "trunk") -> int:
    """The conditioning-feature dimension for a given CVAE checkpoint + feature tap."""
    meta = json.loads((Path(ckpt_dir) / "meta.json").read_text())
    if feature == "trunk":
        return 256
    if feature == "delta":
        return int(meta["n_out"])
    if feature == "raw":
        return len(meta["sensors"])
    raise ValueError(f"unknown prox feature {feature!r} (expected 'trunk', 'delta' or 'raw')")


class ProxCVAEEncoder(nn.Module):
    """Frozen Safety-CVAE used as a skin -> conditioning-feature extractor for ACT.

    forward(prox) takes RAW proximity depths (B, 40, 8, 8) in meters, featurizes them to
    closeness internally (exactly as the CVAE was trained), and returns the conditioning
    feature shaped (B, 1, feat_dim) — ready to pass as ACT's ``proximity_positions`` with
    ``n_proximity_sensors = 1`` (one global skin feature, expanded to K encoder tokens by
    the ACT-side projection).
    """

    def __init__(self, ckpt_dir: str | Path = DEFAULT_CKPT,
                 feature: str = "trunk", device: str = "cuda") -> None:
        super().__init__()
        ckpt_dir = Path(ckpt_dir)
        meta = json.loads((ckpt_dir / "meta.json").read_text())
        if feature not in ("trunk", "delta", "raw"):
            raise ValueError(f"unknown prox feature {feature!r} (expected 'trunk'/'delta'/'raw')")
        self.feature = feature
        self.label_scale = float(meta["label_scale"])
        self.sensor_order: list[str] = list(meta["sensors"])
        self.n_sensors = len(self.sensor_order)
        self.n_out = int(meta["n_out"])
        self.feat_dim = {"trunk": 256, "delta": self.n_out, "raw": self.n_sensors}[feature]

        model = SafetyCVAE(int(meta["n_in"]), int(meta["n_out"]), int(meta["z_dim"]))
        state = torch.load(ckpt_dir / "model.pt", map_location=device)
        model.load_state_dict(state)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model
        self.device = device
        self.to(device)
        print(
            f"[prox] frozen Safety-CVAE from {ckpt_dir} | feature={feature} "
            f"feat_dim={self.feat_dim} | {self.n_sensors} sensors | label_scale={self.label_scale:.3f}"
        )

    @torch.no_grad()
    def forward(self, prox: torch.Tensor) -> torch.Tensor:
        """(B, 40, 8, 8) raw depths (m) -> (B, 1, feat_dim) conditioning feature."""
        if prox.dim() != 4 or prox.shape[1] != self.n_sensors:
            raise ValueError(
                f"prox must be (B, {self.n_sensors}, 8, 8); got {tuple(prox.shape)}"
            )
        prox = prox.to(self.device)
        x = featurize_torch(prox)                       # (B, 2560)
        if self.feature == "trunk":
            feat = self.model.trunk(x)                  # (B, 256)
        elif self.feature == "raw":
            # per-sensor peak closeness, already in [0, 1] — bypasses the CVAE entirely
            # (identical math to scripts/probe_prox_decodability.py raw_skin_feature)
            feat = x.reshape(x.shape[0], self.n_sensors, -1).amax(dim=2)  # (B, 40)
        else:
            feat = self.model.act(x) * self.label_scale  # (B, 7) real joint units
        return feat.unsqueeze(1)                         # (B, 1, feat_dim)


def stack_obs_proximity(obs: dict, sensor_order: list[str]) -> np.ndarray:
    """Build a (40, 8, 8) float32 depth array from a live env observation dict.

    Each ``obs[sensor_name]`` is the ProximityDepthBufferSensor reading, shape
    (n_substeps, 8, 8) (or (8, 8) if un-substepped) in meters. Substeps are mean-pooled,
    matching the dataset converter. Sensors are stacked in ``sensor_order`` (the CVAE's
    meta.json order). Raises a clear error listing available keys if a sensor is missing.
    """
    frames = []
    for name in sensor_order:
        if name not in obs:
            avail = [k for k in obs if "sensor" in str(k)]
            raise KeyError(
                f"proximity sensor {name!r} not in observation. "
                f"{len(avail)} sensor-like keys present, e.g. {avail[:6]}. "
                f"Is the hybrid-skin ProximityDepthBufferSensor enabled?"
            )
        arr = np.asarray(obs[name], dtype=np.float32)
        if arr.ndim == 3:          # (n_substeps, 8, 8) -> mean-pool substeps
            arr = arr.mean(axis=0)
        elif arr.ndim != 2:
            raise ValueError(f"sensor {name!r} has unexpected shape {arr.shape}")
        frames.append(arr)         # (8, 8)
    return np.stack(frames, axis=0)  # (40, 8, 8)
