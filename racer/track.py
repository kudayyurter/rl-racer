"""Track model: a closed centerline spline sampled evenly, plus walls, projection and pose lookup.

Why this exists: everything else stands on the track. The simulator uses the walls for
crashes and range sensors, the projection for the progress reward and lap counting, and the
renderer draws it. A track file only lists a few control points; this module turns them
into a smooth loop and precomputes everything the hot simulation loop needs as numpy arrays.

Conventions: world y-up, meters. Control points are listed in driving direction, so the
counter-clockwise order means driving counter-clockwise. ``normal`` points to the left of
the tangent, so ``left`` is the wall on the driver's left (the inner wall of a CCW corner).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Tracks live next to the package (repo_root/tracks), found from this file, not the CWD.
TRACKS_DIR = Path(__file__).resolve().parent.parent / "tracks"

# Dense points per control-point span used for the first pass of the spline. Dense enough
# that re-sampling by chord length is accurate to well under a millimeter.
_DENSE_PER_SPAN = 200


def catmull_rom_closed(points: np.ndarray, n_samples: int) -> np.ndarray:
    """Closed centripetal (alpha 0.5) Catmull-Rom spline through `points`, (P,2) -> (n_samples,2).

    Centripetal parameterization avoids the loops and cusps the uniform variant makes when
    control points are unevenly spaced. The curve is first evaluated densely, then
    re-sampled so the returned points are evenly spaced by arc length (the dense spans have
    uneven speed, and the simulator wants uniform samples).
    """
    p = np.asarray(points, dtype=float)
    count = len(p)
    u = np.linspace(0.0, 1.0, _DENSE_PER_SPAN, endpoint=False)[:, None]  # (D,1)

    spans = []
    for i in range(count):
        p0, p1, p2, p3 = (p[(i + k) % count] for k in (-1, 0, 1, 2))
        # Knot values: each step is the chord length to the power alpha = 0.5.
        t0 = 0.0
        t1 = t0 + np.linalg.norm(p1 - p0) ** 0.5
        t2 = t1 + np.linalg.norm(p2 - p1) ** 0.5
        t3 = t2 + np.linalg.norm(p3 - p2) ** 0.5
        t = t1 + u * (t2 - t1)
        # Barry-Goldman pyramid: blend neighbours linearly, then blend the blends.
        a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
        b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
        spans.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    dense = np.concatenate(spans)  # (P*D, 2), closed loop without repeating the first point

    # Arc length along the dense loop (including the closing step back to the start).
    closed = np.vstack([dense, dense[:1]])
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    targets = np.linspace(0.0, cum[-1], n_samples, endpoint=False)
    return np.stack([np.interp(targets, cum, closed[:, 0]),
                     np.interp(targets, cum, closed[:, 1])], axis=1)


@dataclass(frozen=True)
class Track:
    """A sampled closed track. All arrays are indexed by sample 0..M-1 (don't mutate them).

    Wall segment ``i`` of each wall runs from sample ``i`` to sample ``(i + 1) % M``.
    """

    name: str
    width: float
    center: np.ndarray    # (M,2) centerline samples
    tangent: np.ndarray   # (M,2) unit, driving direction
    normal: np.ndarray    # (M,2) unit, left of the tangent
    left: np.ndarray      # (M,2) left wall = center + normal * width/2
    right: np.ndarray     # (M,2) right wall = center - normal * width/2
    cum_len: np.ndarray   # (M,) arc length at each sample, cum_len[0] = 0
    length: float         # total loop length in meters
    start_index: int      # sample where cars start (the start line)
    spacing: float        # nominal meters between samples (length / M)

    def wall_window(self, idx: np.ndarray, half: int) -> tuple[np.ndarray, np.ndarray]:
        """Wall segment endpoints near each car: both walls, samples idx-half .. idx+half.

        Returns (a, b), each (N,K,2) with K = 2*(2*half+1): the left wall's segments first,
        then the right wall's. Segment k runs a[k] -> b[k]. Windowing keeps sensors and
        crash checks cheap: only walls near the car can matter.
        """
        m = len(self.center)
        offsets = np.arange(-half, half + 1)
        seg = (np.asarray(idx)[:, None] + offsets[None, :]) % m   # (N, 2*half+1)
        nxt = (seg + 1) % m
        a = np.concatenate([self.left[seg], self.right[seg]], axis=1)
        b = np.concatenate([self.left[nxt], self.right[nxt]], axis=1)
        return a, b

    def project(self, pos: np.ndarray, hint: np.ndarray, half: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Where is each car along the track?

        Searches only samples within hint +- half (wrapping). A global nearest-point search
        could jump to another part of the track that passes close by (the far side of a
        hairpin); trusting the previous index prevents that.

        Returns (idx, s): the nearest sample index, and the arc length of the car's
        projection onto the centerline segment next to that sample (the one on either side
        with the smaller distance). Projecting onto the segment, not just taking the sample's
        s, makes s continuous instead of quantized to the sample spacing. s is in [0, length).
        """
        m = len(self.center)
        pos = np.asarray(pos, dtype=float)
        cand = (np.asarray(hint)[:, None] + np.arange(-half, half + 1)[None, :]) % m    # (N,K)
        d2 = ((self.center[cand] - pos[:, None, :]) ** 2).sum(axis=2)
        idx = cand[np.arange(len(pos)), d2.argmin(axis=1)]

        best_dist = np.full(len(pos), np.inf)
        best_s = np.zeros(len(pos))
        for start in ((idx - 1) % m, idx):  # segment before the sample, segment after it
            a = self.center[start]
            b = self.center[(start + 1) % m]
            ab = b - a
            seg_len2 = (ab ** 2).sum(axis=1)
            frac = np.clip(((pos - a) * ab).sum(axis=1) / seg_len2, 0.0, 1.0)
            dist = np.linalg.norm(pos - (a + frac[:, None] * ab), axis=1)
            # The last segment closes the loop, so its length is length - cum_len[M-1].
            seg_len = np.where(start == m - 1, self.length - self.cum_len[-1], np.sqrt(seg_len2))
            better = dist < best_dist
            best_dist = np.where(better, dist, best_dist)
            best_s = np.where(better, self.cum_len[start] + frac * seg_len, best_s)
        return idx, best_s % self.length

    def pose_at(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Centerline position (N,2) and heading in radians (N,) at arc length s (mod length).

        Position is linear between samples; heading comes from the interpolated tangent,
        so it turns smoothly instead of stepping at every sample.
        """
        m = len(self.center)
        s = np.asarray(s, dtype=float) % self.length
        i = np.clip(np.searchsorted(self.cum_len, s, side="right") - 1, 0, m - 1)
        j = (i + 1) % m
        seg_len = np.where(i == m - 1, self.length - self.cum_len[-1], self.cum_len[(i + 1) % m] - self.cum_len[i])
        frac = ((s - self.cum_len[i]) / seg_len)[:, None]
        pos = self.center[i] + frac * (self.center[j] - self.center[i])
        tan = self.tangent[i] + frac * (self.tangent[j] - self.tangent[i])
        return pos, np.arctan2(tan[:, 1], tan[:, 0])

    def bounds(self) -> tuple[float, float, float, float]:
        """(xmin, ymin, xmax, ymax) over both walls, for fitting the camera."""
        walls = np.vstack([self.left, self.right])
        xmin, ymin = walls.min(axis=0)
        xmax, ymax = walls.max(axis=0)
        return float(xmin), float(ymin), float(xmax), float(ymax)


def build_track(name: str, points, width: float, start_index: int = 0, spacing: float = 2.0) -> Track:
    """Turn control points into a Track with samples about `spacing` meters apart.

    `start_index` names a control point; the start line is the sample nearest to it.
    """
    pts = np.asarray(points, dtype=float)
    # First pass: sample coarsely just to measure the length, which sets the sample count.
    approx = catmull_rom_closed(pts, 4 * len(pts) * 20)
    approx_len = np.linalg.norm(np.diff(np.vstack([approx, approx[:1]]), axis=0), axis=1).sum()
    m = int(round(approx_len / spacing))
    center = catmull_rom_closed(pts, m)

    steps = np.diff(np.vstack([center, center[:1]]), axis=0)   # (M,2), step i: sample i -> i+1
    step_len = np.linalg.norm(steps, axis=1)
    cum_len = np.concatenate([[0.0], np.cumsum(step_len)[:-1]])
    length = float(step_len.sum())

    # Central-difference tangent: smoother than one-sided, and walls inherit its normal.
    tangent = np.roll(center, -1, axis=0) - np.roll(center, 1, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)   # rotate +90 degrees = left

    half = width / 2.0
    start = int(np.linalg.norm(center - pts[start_index], axis=1).argmin())
    return Track(name=name, width=float(width), center=center, tangent=tangent, normal=normal,
                 left=center + normal * half, right=center - normal * half, cum_len=cum_len,
                 length=length, start_index=start, spacing=length / m)


def load_track(path_or_name: str) -> Track:
    """Load a track from JSON. A bare name like "twisty" means tracks/twisty.json in the repo.

    Anything containing a path separator, or ending in .json, is treated as a file path.
    """
    is_path = "/" in path_or_name or "\\" in path_or_name or path_or_name.endswith(".json")
    path = Path(path_or_name) if is_path else TRACKS_DIR / f"{path_or_name}.json"
    data = json.loads(path.read_text())
    return build_track(data["name"], data["points"], data["width"], data.get("start_index", 0))
