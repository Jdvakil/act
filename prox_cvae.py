"""PACT import path. Canonical code: encoders/peak_closeness.py.

Train and eval still ``import prox_cvae``. Do not add logic here.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from encoders.peak_closeness import (  # noqa: E402, F401
    DEFAULT_CKPT,
    D_MAX,
    DEAD_PIXEL_M,
    HYBRID_SKIN_SENSOR_ORDER,
    PeakClosenessEncoder,
    ProxCVAEEncoder,
    SafetyCVAE,
    feat_dim_for,
    featurize_np,
    featurize_torch,
    resolve_prox_layout,
    stack_obs_proximity,
)

__all__ = [
    "DEFAULT_CKPT",
    "D_MAX",
    "DEAD_PIXEL_M",
    "HYBRID_SKIN_SENSOR_ORDER",
    "PeakClosenessEncoder",
    "ProxCVAEEncoder",
    "SafetyCVAE",
    "feat_dim_for",
    "featurize_np",
    "featurize_torch",
    "resolve_prox_layout",
    "stack_obs_proximity",
]
