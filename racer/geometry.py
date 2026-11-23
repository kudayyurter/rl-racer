"""Vectorized 2D geometry for the simulator: ray-to-wall distances and segment crossings.

Why this exists: the car's range sensors and crash detection run every physics
step for every car at once. Both reduce to "does this segment/ray hit that
segment?", so they are written once with numpy broadcasting (no Python loops
over cars, rays or walls).

Both functions use the same 2D cross-product formulation. For a ray with origin
``o`` and direction ``d`` hitting a segment from ``a`` to ``b`` (``s = b - a``)::

    o + t*d = a + u*s   =>   t = cross(a-o, s) / cross(d, s)
                             u = cross(a-o, d) / cross(d, s)

The hit is real when ``t >= 0`` (in front of the ray) and ``0 <= u <= 1``
(within the segment). ``cross(d, s) == 0`` means parallel: there is no unique
intersection, so we treat it as "no hit".
"""

import numpy as np

# Below this |cross(d, s)| the two directions are treated as parallel.
PARALLEL_EPS = 1e-9


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """2D cross product (the z component) over the last axis, shape (..., 2) -> (...)."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _safe_divide(numer: np.ndarray, denom: np.ndarray, parallel: np.ndarray) -> np.ndarray:
    """numer / denom without ever dividing by ~0 (so no warnings).

    The value is arbitrary where ``parallel`` is True; callers must mask those entries.
    """
    return numer / np.where(parallel, 1.0, denom)


def ray_distances(origins: np.ndarray, dirs: np.ndarray, seg_a: np.ndarray,
                  seg_b: np.ndarray, max_range: float) -> np.ndarray:
    """Distance from each ray to the nearest wall segment.

    origins (N,2), dirs (N,R,2) unit vectors, seg_a/seg_b (N,K,2) wall endpoints.
    Returns (N,R) in [0, max_range]; ``max_range`` where nothing is hit in range.

    Parallel rays never hit. That includes the collinear case (wall lying on the
    ray's own line): the true answer is a whole interval, so we pick the
    simplest finite answer, "no hit" = max_range. Walls are closed polylines in
    practice, so a ray running along one also meets the adjoining wall.
    """
    o = origins[:, None, None, :]        # (N,1,1,2)
    d = dirs[:, :, None, :]              # (N,R,1,2)
    a = seg_a[:, None, :, :]             # (N,1,K,2)
    s = (seg_b - seg_a)[:, None, :, :]   # (N,1,K,2)

    denom = _cross(d, s)                 # (N,R,K)
    parallel = np.abs(denom) < PARALLEL_EPS
    ao = a - o                           # (N,1,K,2)
    t = _safe_divide(_cross(ao, s), denom, parallel)  # distance along the ray (d is unit)
    u = _safe_divide(_cross(ao, d), denom, parallel)  # position along the wall, 0..1

    hit = ~parallel & (t >= 0.0) & (u >= 0.0) & (u <= 1.0)
    t = np.where(hit, t, max_range)
    return np.minimum(t.min(axis=2), max_range)


def any_crossing(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Per car: does any segment p1->p2 cross any segment q1->q2?

    p1/p2 (N,E,2), q1/q2 (N,K,2). Returns (N,) bool. Touching endpoints count as
    crossing (both parameters use closed intervals). Parallel pairs, including
    collinear overlaps, are not reported; callers use short segments where that
    case does not matter. Excluding segments that should not count (e.g. a
    polyline's own neighbours) is the caller's job.
    """
    p = p1[:, :, None, :]                # (N,E,1,2)
    r = (p2 - p1)[:, :, None, :]
    q = q1[:, None, :, :]                # (N,1,K,2)
    s = (q2 - q1)[:, None, :, :]

    denom = _cross(r, s)                 # (N,E,K)
    parallel = np.abs(denom) < PARALLEL_EPS
    qp = q - p
    t = _safe_divide(_cross(qp, s), denom, parallel)  # position along p, 0..1
    u = _safe_divide(_cross(qp, r), denom, parallel)  # position along q, 0..1

    crossing = ~parallel & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)
    return crossing.any(axis=(1, 2))
