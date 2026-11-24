"""Lesson 01 experiment: how does the sample spacing change projection accuracy and speed?

A track is stored as centerline samples about ``spacing`` meters apart, and everything (walls,
progress, sensors) uses the straight segments between samples. This script builds each shipped
track at two spacings and compares them with a very fine reference (0.25 m spacing) that stands
in for the true spline.

For each track and spacing it prints:

* samples     how many centerline samples the track has
* length      loop length measured along the samples (chords cut corners, so coarser = shorter)
* min R       tightest corner radius of the sampled centerline (three-point circles, 6 m apart)
* chord gap   largest distance from the true curve to the sampled polyline; theory spacing^2 / 8R
* s error     largest error in lap progress (meters) for a point moving along the true curve,
              on the centerline (d=0) and 4 m left of it (d=4). Progress is compared as a lap
              fraction times the reference length, so the different total lengths do not count
* step error  largest error in the progress gained over one physics substep (0.83 m of travel,
              50 m/s at 1/60 s), which is what one substep's progress reward sees
* car-steps/s simulator throughput with 256 cars (2 s of random driving), because the sensor
              rays check every wall segment within ray range and finer spacing means more of them

Run: uv run python docs/lessons/scripts/projection_error.py   (about 30 s)
"""

import json
import time

import numpy as np

from racer.config import EnvConfig
from racer.sim import BatchSim
from racer.track import TRACKS_DIR, Track, build_track

SPACINGS = (2.0, 8.0)
REFERENCE_SPACING = 0.25
STEP_M = 50.0 / 60.0          # top speed times one physics step
OFFSETS = (0.0, 4.0)          # meters left of the centerline


def min_radius(track: Track, span_m: float = 6.0) -> float:
    """Smallest radius of the circle through three samples about span_m apart."""
    k = max(1, round(span_m / track.spacing))
    p0, p1, p2 = (np.roll(track.center, -s, axis=0) for s in (-k, 0, k))
    a = np.linalg.norm(p1 - p0, axis=1)
    b = np.linalg.norm(p2 - p1, axis=1)
    c = np.linalg.norm(p2 - p0, axis=1)
    twice_area = np.abs((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
                        - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0]))
    with np.errstate(divide="ignore"):
        return float((a * b * c / (2.0 * twice_area)).min())


def chord_gap(coarse: Track, ref: Track) -> float:
    """Largest distance from a reference point to the nearest coarse centerline segment."""
    a = coarse.center
    b = np.roll(coarse.center, -1, axis=0)
    worst = 0.0
    for chunk in np.array_split(ref.center, 20):           # chunks keep memory small
        ab = b - a                                           # (M,2)
        ap = chunk[:, None, :] - a[None, :, :]               # (C,M,2)
        frac = np.clip((ap * ab).sum(-1) / (ab ** 2).sum(-1), 0.0, 1.0)
        dist = np.linalg.norm(ap - frac[..., None] * ab, axis=-1).min(axis=1)
        worst = max(worst, float(dist.max()))
    return worst


def progress_errors(coarse: Track, ref: Track, offset: float) -> tuple[float, float]:
    """(max s error, max per-step error) in meters for a point riding the reference path."""
    s_ref = np.arange(0.0, ref.length, STEP_M)
    pos, _ = ref.pose_at(s_ref)
    i = np.clip(np.searchsorted(ref.cum_len, s_ref, side="right") - 1, 0, len(ref.center) - 1)
    pos = pos + ref.normal[i] * offset
    s_c, s_r = [], []
    hint_c = np.array([0])
    hint_r = np.array([0])
    for p in pos:                       # one projection per substep, as the simulator does
        hint_c, sc = coarse.project(p[None], hint_c)
        hint_r, sr = ref.project(p[None], hint_r, half=40)
        s_c.append(sc[0])
        s_r.append(sr[0])
    # Unwrap the loop so s keeps growing, then compare as lap fractions in reference meters.
    frac_c = np.unwrap(np.array(s_c) / coarse.length * 2 * np.pi) / (2 * np.pi)
    frac_r = np.unwrap(np.array(s_r) / ref.length * 2 * np.pi) / (2 * np.pi)
    err = (frac_c - frac_r) * ref.length
    err -= err[0]                       # both start at the same point
    step_err = np.diff(frac_c * ref.length) - np.diff(frac_r * ref.length)
    return float(np.abs(err).max()), float(np.abs(step_err).max())


def throughput(track: Track, n_cars: int = 256, seconds: float = 2.0) -> float:
    sim = BatchSim(track, EnvConfig(), n_cars, seed=0, auto_reset=True)
    sim.reset()
    rng = np.random.default_rng(0)
    sim.step(rng.uniform(-1, 1, (n_cars, 2)))
    decisions, start = 0, time.perf_counter()
    while time.perf_counter() - start < seconds:
        sim.step(rng.uniform(-1, 1, (n_cars, 2)))
        decisions += 1
    return decisions * n_cars / (time.perf_counter() - start)


def main() -> None:
    header = (f"{'track':7} {'spacing':>7} {'samples':>7} {'length':>8} {'min R':>6} "
              f"{'chord gap':>9} {'s err d=0':>9} {'s err d=4':>9} {'step err d=4':>12} "
              f"{'car-steps/s':>11}")
    print(header)
    for name in ("oval", "twisty"):
        data = json.loads((TRACKS_DIR / f"{name}.json").read_text())
        ref = build_track(name, data["points"], data["width"], data.get("start_index", 0),
                          spacing=REFERENCE_SPACING)
        for spacing in SPACINGS:
            t = build_track(name, data["points"], data["width"], data.get("start_index", 0),
                            spacing=spacing)
            errs = {d: progress_errors(t, ref, d) for d in OFFSETS}
            print(f"{name:7} {spacing:7.1f} {len(t.center):7d} {t.length:8.2f} "
                  f"{min_radius(t):6.1f} {chord_gap(t, ref):9.3f} {errs[0.0][0]:9.3f} "
                  f"{errs[4.0][0]:9.3f} {errs[4.0][1]:12.3f} {throughput(t):11,.0f}")
        print(f"{name:7} {'ref':>7} {len(ref.center):7d} {ref.length:8.2f} {min_radius(ref):6.1f}")


if __name__ == "__main__":
    main()
