"""Lesson 09: drive trained checkpoints on the 10 held-out random tracks and on twisty.

First prints the tracks themselves, for each held-out track (seeds 10_000..10_009, the set
``racer/algos/ppo.py`` logs as ``eval_heldout_laps_mean``) and for twisty and oval: length,
tightest corner radius (the local measure ``track_is_valid`` uses), the sharpest turn (the
largest heading change within any 60 m of road), and the right-hand share (how much of all the
turning the road does is to the right). Radius and sharpest turn differ: a short kink can be as
tight as a hairpin locally, but only a hairpin turns the car round. Together they let "laps on
held-out" and "laps on twisty" be compared for difficulty.

Then, for each checkpoint given, one deterministic episode per track with ``racer.evaluate`` (the
same evaluation training logs) and the run's own env config: laps, best lap, meters driven, and
how the episode ended (``2 laps``, ``time limit``, ``crash`` or ``wrong way``). The summary line
gives the mean laps over the held-out set, which for a random-track run must match the
``eval_heldout_laps_mean`` it logged at that iteration (a check that this script measures the
same thing).

``--mirror`` drives every track (held-out and twisty) mirrored instead: y -> -y, so each left
bend becomes the same bend to the right and the loop runs clockwise. Every track the project
trains on runs counter-clockwise (random tracks by construction, twisty as drawn), so a
mirrored track asks whether a policy learned "corners" or "mostly left corners".

Arguments: ``RUN`` (checkpoints/latest.pt, the final evaluation), ``RUN@ITERATION``
(checkpoints/ckpt_<iteration>.pt), or ``RUN@best``: the checkpoint whose twisty evaluation
(logged in metrics.csv) was best, ranked by most laps, then fastest lap, then most meters. For
a run that lapped twisty that is its fastest-lap checkpoint; for one that never did, the one
that got furthest. The choice looks only at twisty, never at the held-out tracks, so the
held-out numbers stay a clean test, and the same rule applies to twisty-only and random-track
runs alike.

Run: uv run python docs/lessons/scripts/heldout_eval.py runs/ppo-<ts>-s0 runs/ppo-<ts>-s0@best
     uv run python docs/lessons/scripts/heldout_eval.py --mirror runs/ppo-<ts>-s0
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from racer.config import EnvConfig, from_dict
from racer.evaluate import evaluate
from racer.policy import policy_from_checkpoint
from racer.runlog import RunDir
from racer.sim import BatchSim
from racer.track import TRACKS_DIR, Track, build_track, load_track
from racer.trackgen import (HELDOUT_FIRST_SEED, _draw_track, _min_corner_radius, heldout_tracks,
                            track_for_seed, training_seed)

N_HELDOUT = 10  # the same set ppo.py evaluates on in random-track mode
N_TRAINING = 200  # training tracks summarized: the first ones a seed-0 run drives
TURN_WINDOW_M = 60.0  # road length over which "sharpest turn" is measured (a hairpin fits in it)


def checkpoint_path(arg: str) -> tuple[Path, Path]:
    """(run directory, checkpoint file) for RUN, RUN@ITERATION or RUN@best."""
    run, _, which = arg.partition("@")
    run = Path(run)
    if not which:
        return run, run / "checkpoints" / "latest.pt"
    if which == "best":
        evals = [r for r in RunDir.open(run).read_metrics() if not math.isnan(r["eval_laps"])]

        def rank(r):  # smaller is better: more laps, then a faster lap, then more meters
            lap = r["eval_best_lap_s"]
            return (-r["eval_laps"], math.inf if math.isnan(lap) else lap, -r["eval_progress_m"])

        which = str(int(min(evals, key=rank)["iteration"]))
    return run, run / "checkpoints" / f"ckpt_{int(which):05d}.pt"


def ending(policy, track: Track, cfg: EnvConfig) -> str:
    """Replay the same deterministic episode and say how it ended (evaluate does not record it)."""
    sim = BatchSim(track, cfg, n_cars=1, seed=0, auto_reset=False)
    obs, end = sim.reset(), ""
    while sim.alive[0]:
        res = sim.step(policy.act(obs))
        obs = res.obs
        if res.terminated[0]:
            # A crash pays the penalty; the wrong-way rule terminates without one.
            end = "crash" if res.reward[0] < -cfg.crash_penalty / 2 else "wrong way"
        elif res.truncated[0]:
            end = f"{cfg.max_laps} laps" if int(sim.laps[0]) >= cfg.max_laps else "time limit"
    return end


def sharpest_turn_deg(track: Track, window_m: float = TURN_WINDOW_M) -> float:
    """Largest change of road heading (degrees) within any ``window_m`` of centerline."""
    step = np.roll(track.center, -1, axis=0) - track.center
    heading = np.unwrap(np.arctan2(step[:, 1], step[:, 0]))
    # Go round the loop twice so windows that cross the start line are measured too. One lap of
    # a closed loop turns the heading by exactly one full turn (+2 pi counter-clockwise).
    full_turn = 2.0 * np.pi * np.sign(heading[-1] - heading[0])
    heading = np.concatenate([heading, heading + full_turn])
    s = np.concatenate([track.cum_len, track.cum_len + track.length])
    best = 0.0
    for i in range(len(track.center)):
        j = np.searchsorted(s, s[i] + window_m)
        best = max(best, float(np.abs(heading[i:j + 1] - heading[i]).max()))
    return float(np.degrees(best))


def mirrored(name: str, points, width: float, start_index: int = 0) -> Track:
    """The same road with left and right swapped (y -> -y): every bend turns the other way."""
    return build_track(f"{name}_mirror", np.asarray(points, dtype=float) * [1.0, -1.0], width,
                       start_index)


def mirrored_tracks() -> list[Track]:
    """The held-out set and twisty, mirrored. Rebuilt from their control points."""
    tracks = []
    for i in range(N_HELDOUT):
        seed = HELDOUT_FIRST_SEED + i
        points, track = _draw_track(np.random.default_rng(seed))   # what track_for_seed draws
        tracks.append(mirrored(f"heldout_{seed}", points, track.width))
    data = json.loads((TRACKS_DIR / "twisty.json").read_text())
    tracks.append(mirrored("twisty", data["points"], data["width"], data.get("start_index", 0)))
    return tracks


def right_share(track: Track) -> float:
    """Share of the road's total turning (left plus right, degrees) that bends to the right."""
    step = np.roll(track.center, -1, axis=0) - track.center
    heading = np.arctan2(step[:, 1], step[:, 0])
    turn = np.angle(np.exp(1j * (np.roll(heading, -1) - heading)))  # wrapped to (-pi, pi]
    return float(-turn[turn < 0].sum() / np.abs(turn).sum())


def describe(track: Track) -> str:
    return (f"{track.name:>20}: {track.length:7.1f} m, tightest corner radius "
            f"{_min_corner_radius(track):5.1f} m, sharpest turn {sharpest_turn_deg(track):4.0f} deg "
            f"in {TURN_WINDOW_M:.0f} m, {100 * right_share(track):3.0f}% of turning to the right")


def main() -> None:
    args = sys.argv[1:]
    mirror = "--mirror" in args
    args = [a for a in args if a != "--mirror"]
    if mirror:
        *heldout, twisty = mirrored_tracks()
        shown = heldout + [twisty]
    else:
        heldout, twisty = heldout_tracks(N_HELDOUT), load_track("twisty")
        shown = heldout + [twisty, load_track("oval")]
    print("tracks" + (" (mirrored)" if mirror else ""))
    for track in shown:
        print("  " + describe(track))
    lengths = [t.length for t in heldout]
    print(f"  held-out lengths {min(lengths):.1f} .. {max(lengths):.1f} m")
    # The same seed sequence train_ppo draws with --seed 0 (np.random.default_rng(seed)).
    rng = np.random.default_rng(0)
    training = [track_for_seed(training_seed(rng, N_HELDOUT)) for _ in range(N_TRAINING)]
    lengths = [t.length for t in training]
    turns = [sharpest_turn_deg(t) for t in training]
    radii = [_min_corner_radius(t) for t in training]
    rights = [100 * right_share(t) for t in training]
    print(f"  first {N_TRAINING} training tracks of a seed-0 run: lengths {min(lengths):.1f} .. "
          f"{max(lengths):.1f} m, tightest corner radius {min(radii):.1f} .. {max(radii):.1f} m, "
          f"sharpest turn {min(turns):.0f} .. {max(turns):.0f} deg in {TURN_WINDOW_M:.0f} m, "
          f"{min(rights):.0f} .. {max(rights):.0f}% of turning to the right")

    for arg in args:
        run, path = checkpoint_path(arg)
        ckpt = torch.load(path, weights_only=False)
        policy = policy_from_checkpoint(ckpt, path)
        config = json.loads((run / "config.json").read_text())
        cfg = from_dict(EnvConfig, config["env"])
        trained_on = config["train"].get("tracks", "fixed")
        trained_on = "random tracks" if trained_on == "random" else config["train"]["track"]
        print(f"\n{arg}  (trained on {trained_on}, iteration {ckpt['iteration']}, "
              f"{ckpt['env_steps']:,} steps)")
        laps = []
        for track in heldout + [twisty]:
            res = evaluate(policy, track, cfg)
            best = f"{res.best_lap_s:6.2f} s" if res.laps else "     -  "
            print(f"  {track.name:>20}: {res.laps} laps, best lap {best}, "
                  f"{res.progress_m:7.1f} m, {ending(policy, track, cfg)}")
            if track is not twisty:
                laps.append(res.laps)
        print(f"  held-out: mean laps {sum(laps) / len(laps):.2f}, "
              f"{sum(n >= 1 for n in laps)}/{len(laps)} tracks lapped at least once, "
              f"{sum(n >= cfg.max_laps for n in laps)}/{len(laps)} drove all {cfg.max_laps} laps")


if __name__ == "__main__":
    main()
