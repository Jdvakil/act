"""V10.10 / v12 scene contract for pact_pick_n_place_v2 eval. No molmospaces import.

XMLs live in repo ``custom_scenes/``: hashed ``v10_7_{neg5,center,pos5}`` plus
the include chain ``v5.xml`` → ``v3.xml``. ``pact_place_corridor_v12.xml``
includes the local center file. Sampler still hashes the three v10_7 files;
the v12 wrapper is never a sampler path.

Train dump is ``pact_place_corridor_v10_11d_randomized_clutter``.
Molmospaces Python on ``origin/main`` only has
``PactPlaceCorridorV1010FourObjectSampler`` — that is OOD vs collect.
V1011D lives on ``70dedc0``.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
CUSTOM_SCENES = _REPO_ROOT / "custom_scenes"
V12_XML = CUSTOM_SCENES / "pact_place_corridor_v12.xml"
V12_INCLUDE_STEM = "pact_place_corridor_v10_7_center.xml"
MOLMOSPACES_V1010_SHA = "4bba4cbcea49ca8dbaee44fb9a376568b1b3cc82"
DEFAULT_WORKTREE = Path("/home/jaydv/code/molmospaces-pact-v1010")
INCLUDE_CHAIN = (
    "pact_place_corridor_v5.xml",
    "pact_place_corridor_v3.xml",
)

LAYOUT_FAMILIES = (
    "F0_target_side_stagger",
    "F1_inner_panel_stagger",
    "F2_outer_panel_stagger",
    "F3_aperture_side_stagger",
)
INTRUSION_SIDES = ("left", "right")
POSE_IDS = ("neg5", "center", "pos5")
N_V1010_CELLS = 24

# Frozen hashes from molmospaces origin/main pact_place/contracts.py (V10.10).
V1010_SCENE_BY_POSE = {
    "neg5": {
        "filename": "pact_place_corridor_v10_7_neg5.xml",
        "sha256": "df50679c749c6ad771d00023e73a08e0bfaf59d5391df9b42cf05de4ed7893a7",
    },
    "center": {
        "filename": "pact_place_corridor_v10_7_center.xml",
        "sha256": "b5a41d0d8934240b078f1cdbf3a6991b2e94a46558ddf1c9eae0119c8b8e138a",
    },
    "pos5": {
        "filename": "pact_place_corridor_v10_7_pos5.xml",
        "sha256": "762a5a4662a8fc0d31a3a0ee1135b347d6dd2c882daf4e65c2f706ab2d6fe565",
    },
}

_INCLUDE_RE = re.compile(
    r'<include\s+file="([^"]*pact_place_corridor_v10_7_center\.xml)"\s*/>'
)


def v1010_cell(index: int) -> tuple[str, str, str]:
    cells = [
        (family, side, pose)
        for family in LAYOUT_FAMILIES
        for side in INTRUSION_SIDES
        for pose in POSE_IDS
    ]
    return cells[int(index) % len(cells)]


def v1010_scenes_dir(molmospaces_root: Path) -> Path:
    return (
        Path(molmospaces_root).resolve()
        / "molmo_spaces"
        / "data_generation"
        / "custom_scenes"
    )


def assert_v1010_include_chain(scenes_dir: Path) -> None:
    missing = [
        str(Path(scenes_dir) / name)
        for name in INCLUDE_CHAIN
        if not (Path(scenes_dir) / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "[act-eval-pact-v2] v10_7 include chain missing: " + ", ".join(missing)
        )


def resolve_v1010_scenes_dir(molmospaces_root: Path | None = None) -> Path:
    """Prefer repo ``custom_scenes/`` (hashed v10_7 + v5/v3 chain). Else worktree."""
    try:
        assert_v1010_scene_hashes(CUSTOM_SCENES)
        assert_v1010_include_chain(CUSTOM_SCENES)
        return CUSTOM_SCENES
    except FileNotFoundError:
        pass
    if molmospaces_root is None:
        molmospaces_root = DEFAULT_WORKTREE
    fallback = v1010_scenes_dir(molmospaces_root)
    assert_v1010_scene_hashes(fallback)
    assert_v1010_include_chain(fallback)
    return fallback


def v1010_scene_paths(scenes_dir: Path) -> list[Path]:
    """24 paths, index-aligned with ``v1010_cell`` / the sampler hash check."""
    out: list[Path] = []
    for index in range(N_V1010_CELLS):
        _family, _side, pose = v1010_cell(index)
        out.append(Path(scenes_dir) / V1010_SCENE_BY_POSE[pose]["filename"])
    return out


def assert_v1010_scene_hashes(scenes_dir: Path) -> None:
    missing = []
    mismatched = []
    for pose, meta in V1010_SCENE_BY_POSE.items():
        path = Path(scenes_dir) / meta["filename"]
        if not path.is_file():
            missing.append(str(path))
            continue
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != meta["sha256"]:
            mismatched.append(f"{pose}: {path} {got} != {meta['sha256']}")
    if missing or mismatched:
        parts = []
        if missing:
            parts.append("missing " + ", ".join(missing))
        if mismatched:
            parts.append("hash " + "; ".join(mismatched))
        raise FileNotFoundError(
            "[act-eval-pact-v2] v10_7 scene files: " + "; ".join(parts)
        )


def v12_include_target(v12_xml: Path) -> str:
    text = Path(v12_xml).read_text()
    match = _INCLUDE_RE.search(text)
    if match is None:
        raise ValueError(f"no v10_7_center include in {v12_xml}")
    return match.group(1)


def assert_v12_wraps_center(v12_xml: Path = V12_XML) -> None:
    """7d1ea35 env file must wrap the same center XML the sampler hashes."""
    target = v12_include_target(v12_xml)
    if not target.endswith(V12_INCLUDE_STEM):
        raise ValueError(
            f"[act-eval-pact-v2] {v12_xml} include is {target!r}, "
            f"want *{V12_INCLUDE_STEM}"
        )


def rewrite_v12_include(src: Path, include_xml: Path, dst: Path) -> Path:
    """Rewrite the v12 include to an absolute path. Not a sampler scene file.

    The V1010 sampler hashes file bytes. A rewritten wrapper will never match
    the frozen v10_7 hashes. Use ``v1010_scene_paths`` for eval.
    """
    text = Path(src).read_text()
    new, n = _INCLUDE_RE.subn(
        f'<include file="{Path(include_xml).resolve()}"/>',
        text,
        count=1,
    )
    if n != 1:
        raise ValueError(f"could not rewrite v10_7_center include in {src}")
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(new)
    return dst


def spread_episode_count(num_rollouts: int) -> tuple[int, int]:
    """Return (samples_per_house, total_episodes) for the 24-cell protocol."""
    n = max(1, int(num_rollouts))
    if n < N_V1010_CELLS:
        return 1, N_V1010_CELLS
    per = n // N_V1010_CELLS
    return per, per * N_V1010_CELLS
