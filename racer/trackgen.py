"""Random track generator: closed circuits for training on many tracks instead of one.

Why this exists: a policy trained on one track can memorize it (this corner: steer this much).
Training on a fresh track every rollout and testing on tracks it has never seen shows the
difference between memorizing and learning to drive.

How a track is made: control points sit at sorted angles around a circle, each at a random
radius. Angles increase, so the loop winds once around the center (star-shaped) and can never
cross itself at the control-point level; the spline between points can still bulge or pinch,
so every candidate is checked by ``track_is_valid`` and redrawn if it fails.
"""

import json
from pathlib import Path

import numpy as np

from racer.track import Track, build_track

# Seeds HELDOUT_FIRST_SEED .. +n-1 belong to the evaluation set. Training seeds skip them, so
# the held-out tracks are never seen during training.
HELDOUT_FIRST_SEED = 10_000

# Extra clearance (meters) beyond the road width that two far-apart parts of the road must keep
# between their centerlines, and beyond width/2 that every corner's radius must keep.
_CLEARANCE_M = 4.0
_CORNER_MARGIN_M = 3.0
# "Far apart along the track" means more than this many (road width + clearance) of arc length.
# Nearer samples are neighbours in the same stretch of road and are naturally close.
_FAR_APART_WIDTHS = 3.0
# Curvature is measured over samples this many steps apart (6 m at 2 m spacing), which keeps
# sampling noise from looking like a sharp corner.
_CURVATURE_STRIDE = 3


def _proper_crossings(a1, a2, b1, b2) -> np.ndarray:
    """(M,K) bool: does segment a1[i]->a2[i] cross segment b1[j]->b2[j]? (strict crossings)"""
    def cross(o, p, q):  # sign of the turn o -> p -> q, broadcast over all pairs
        return ((p[..., 0] - o[..., 0]) * (q[..., 1] - o[..., 1])
                - (p[..., 1] - o[..., 1]) * (q[..., 0] - o[..., 0]))

    a1, a2 = a1[:, None, :], a2[:, None, :]
    b1, b2 = b1[None, :, :], b2[None, :, :]
    d1, d2 = cross(a1, a2, b1), cross(a1, a2, b2)   # b's endpoints on opposite sides of a?
    d3, d4 = cross(b1, b2, a1), cross(b1, b2, a2)   # a's endpoints on opposite sides of b?
    return (d1 * d2 < 0) & (d3 * d4 < 0)


def _wall_crosses_itself(wall: np.ndarray) -> bool:
    m = len(wall)
    a1, a2 = wall, np.roll(wall, -1, axis=0)
    hit = _proper_crossings(a1, a2, a1, a2)
    # Neighbouring segments share an endpoint and are not a real crossing: skip them.
    idx = np.arange(m)
    gap = (idx[:, None] - idx[None, :]) % m
    hit &= ~np.isin(gap, (0, 1, m - 1))
    return bool(hit.any())


def _walls_cross_each_other(track: Track) -> bool:
    """Left wall against right wall: catches two sections overlapping side by side."""
    return bool(_proper_crossings(track.left, np.roll(track.left, -1, axis=0),
                                  track.right, np.roll(track.right, -1, axis=0)).any())


def _min_corner_radius(track: Track) -> float:
    """Smallest radius of curvature (m) of the sampled centerline, from three-point circles."""
    k = _CURVATURE_STRIDE
    p0, p1, p2 = (np.roll(track.center, -s, axis=0) for s in (-k, 0, k))
    a = np.linalg.norm(p1 - p0, axis=1)
    b = np.linalg.norm(p2 - p1, axis=1)
    c = np.linalg.norm(p2 - p0, axis=1)
    twice_area = np.abs((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
                        - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0]))
    # circumradius = abc / (4 * area) = abc / (2 * twice_area); collinear points -> infinite
    with np.errstate(divide="ignore"):
        radius = a * b * c / (2.0 * twice_area)
    return float(radius.min())


def _sections_too_close(track: Track) -> bool:
    """Do two parts of the road that are far apart along the track come within asphalt reach?

    If centerlines of distant sections are closer than width + clearance, the asphalt overlaps
    or nearly does, and a car could cut across to the other section (the projection that
    measures progress would follow it).
    """
    c = track.center
    d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
    s = track.cum_len
    along = np.abs(s[:, None] - s[None, :])
    along = np.minimum(along, track.length - along)   # distance the short way round the loop
    far = along > _FAR_APART_WIDTHS * (track.width + _CLEARANCE_M)
    return bool((d[far] < track.width + _CLEARANCE_M).any())


def track_is_valid(track: Track) -> bool:
    """Is this track drivable and unambiguous? Used to filter generated tracks (and in tests).

    Rejects: a wall crossing itself; the left wall crossing the right wall; any corner tighter
    than width/2 + 3 m radius (the inner wall would fold over); and distant stretches of road
    that come closer than one road width + 4 m.
    """
    if _wall_crosses_itself(track.left) or _wall_crosses_itself(track.right):
        return False
    if _walls_cross_each_other(track):
        return False
    if _min_corner_radius(track) < track.width / 2 + _CORNER_MARGIN_M:
        return False
    return not _sections_too_close(track)


def random_track(rng: np.random.Generator, n_points: int = 12,
                 radius: tuple[float, float] = (80.0, 160.0), width: float = 12.0,
                 max_tries: int = 100, name: str = "random") -> Track:
    """Draw a random valid star-shaped closed track (counter-clockwise, start at point 0).

    Candidates are redrawn until ``track_is_valid`` passes and the length is 400-1400 m.
    Raises RuntimeError after ``max_tries`` failures (e.g. an impossible radius range).
    """
    return _draw_track(rng, n_points, radius, width, max_tries, name)[1]


def _draw_track(rng: np.random.Generator, n_points: int = 12,
                radius: tuple[float, float] = (80.0, 160.0), width: float = 12.0,
                max_tries: int = 100, name: str = "random") -> tuple[np.ndarray, Track]:
    """random_track, but also returns the control points (what a track file stores)."""
    step = 2.0 * np.pi / n_points
    for _ in range(max_tries):
        # Evenly spaced angles, each nudged by less than 0.35 of a step: sorted order is kept
        # and no two consecutive points can coincide (NaN in the spline otherwise).
        angles = np.arange(n_points) * step + rng.uniform(-0.35, 0.35, n_points) * step
        radii = rng.uniform(radius[0], radius[1], n_points)
        points = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
        track = build_track(name, points, width)
        if 400.0 <= track.length <= 1400.0 and track_is_valid(track):
            return points, track
    raise RuntimeError(f"no valid track after {max_tries} tries (radius range {radius})")


def track_for_seed(seed: int, name: str | None = None) -> Track:
    """The one random track belonging to ``seed`` (same seed, same track)."""
    return random_track(np.random.default_rng(seed), name=name or f"rand_{seed}")


def heldout_tracks(n: int = 10) -> list[Track]:
    """The fixed evaluation set: tracks for seeds 10_000 .. 10_000 + n - 1, never trained on."""
    return [track_for_seed(HELDOUT_FIRST_SEED + i, f"heldout_{HELDOUT_FIRST_SEED + i}")
            for i in range(n)]


def training_seed(rng: np.random.Generator, n_heldout: int = 10) -> int:
    """Draw a track seed from the run's RNG, redrawing if it lands in the held-out range."""
    while True:
        seed = int(rng.integers(0, 2**31 - 1))
        if not HELDOUT_FIRST_SEED <= seed < HELDOUT_FIRST_SEED + n_heldout:
            return seed


def write_track_json(seed: int, path: str | Path) -> Path:
    """Generate the track for ``seed`` and save it in the format ``load_track`` reads.

    The file holds the control points (full float precision), not the sampled walls, so
    loading rebuilds the identical spline.
    """
    points, track = _draw_track(np.random.default_rng(seed), name=f"rand_{seed}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"name": track.name, "width": track.width, "start_index": 0,
                                "points": points.tolist()}))
    return path
