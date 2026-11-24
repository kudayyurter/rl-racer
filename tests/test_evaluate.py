"""Tests for evaluation (racer/evaluate.py): one deterministic episode, best-lap extraction."""

import math

import numpy as np

from racer.config import EnvConfig
from racer.evaluate import evaluate
from racer.track import load_track
from helpers import ScriptedOvalPolicy

CFG = EnvConfig()
TRACK = load_track("oval")


def test_evaluate_scripted_policy_laps_oval():
    res = evaluate(ScriptedOvalPolicy(CFG, TRACK), TRACK, CFG)
    assert res.laps >= 1
    assert math.isfinite(res.best_lap_s) and 20 < res.best_lap_s < 60
    assert len(res.lap_poses) > 0 and res.lap_poses.shape == (len(res.lap_t), 3)
    assert res.lap_t[0] == 0.0
    # Laps are timed per physics substep, the trace per decision: they agree to one decision.
    assert abs(res.lap_t[-1] - res.best_lap_s) <= CFG.dt * CFG.action_repeat + 1e-9
    assert res.progress_m >= TRACK.length
    assert res.ret > 0


def test_evaluate_crashing_policy_has_no_lap():
    class Straight:
        def act(self, obs):
            return np.tile([0.0, 1.0], (len(obs), 1))

    res = evaluate(Straight(), TRACK, CFG)
    assert res.laps == 0 and math.isnan(res.best_lap_s)
    assert res.lap_poses.shape == (0, 3) and res.lap_t.shape == (0,)
    assert res.progress_m < TRACK.length
