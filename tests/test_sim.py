"""Tests for the batched simulator (racer/sim.py): sensors, reward, laps, termination, resets."""

import dataclasses
import math

import numpy as np
import pytest

from racer.config import EnvConfig
from racer.sim import BatchSim, obs_dim
from racer.track import load_track

CFG = EnvConfig()
TRACK = load_track("oval")
START_S = float(TRACK.cum_len[TRACK.start_index])


def make_sim(n_cars: int = 1, cfg: EnvConfig = CFG, **kwargs) -> BatchSim:
    sim = BatchSim(TRACK, cfg, n_cars, **kwargs)
    sim.reset()
    return sim


def actions(n: int, steer: float = 0.0, throttle: float = 0.0) -> np.ndarray:
    return np.tile([steer, throttle], (n, 1)).astype(float)


def test_obs_shape_and_range():
    sim = BatchSim(TRACK, CFG, 4)
    obs = sim.reset()
    assert obs_dim(CFG) == CFG.n_rays + 3
    assert obs.shape == (4, obs_dim(CFG)) == (4, 12)
    assert np.all(np.isfinite(obs))
    rays = obs[:, :CFG.n_rays]
    assert np.all(rays > 0.0) and np.all(rays <= 1.0)
    assert np.all(np.abs(obs[:, CFG.n_rays:]) <= 1.0)
    assert sim.last_rays.shape == (4, CFG.n_rays)
    # At the start the car points along the track, at rest, wheels straight.
    np.testing.assert_allclose(obs[:, CFG.n_rays:], 0.0, atol=1e-6)


def test_side_rays_see_the_walls():
    # Centered on a straight, the leftmost and rightmost rays (90 degrees off the heading,
    # with the default 180 degree spread) point straight at the walls, width/2 away.
    sim = make_sim()
    np.testing.assert_allclose(sim.last_rays[0, [0, -1]], TRACK.width / 2, rtol=0.02)


def test_driving_forward_earns_progress():
    sim = make_sim()
    total = 0.0
    for _ in range(20):
        r = sim.step(actions(1, throttle=0.5))
        assert not r.terminated[0] and not r.truncated[0]
        total += r.reward[0]
    assert total > 0.0
    assert sim.alive[0]


def test_crash_terminates_with_penalty():
    sim = make_sim()
    sim.place_car(0, s=10.0, lateral=TRACK.width / 2 - 0.5)  # corner already over the left wall
    r = sim.step(np.zeros((1, 2)))
    assert r.terminated[0] and not r.truncated[0]
    assert r.reward[0] <= -CFG.crash_penalty + 1e-6
    assert not sim.alive[0]


def test_lap_counted_once():
    # P-controller: steer against the heading error and the sideways offset from the
    # centerline (normal points left, so a positive offset means "too far left"). Throttle
    # is 0.4 but it coasts above 20 m/s: held at 0.4 the car reaches ~31 m/s on the straight,
    # and the grip limit (12 m/s^2) only allows ~24 m/s round the 50 m end turns.
    sim = make_sim()
    hint = np.array([TRACK.start_index])
    obs = sim.reset()
    lap_decisions = []
    for k in range(1200):  # ~80 s; one lap takes ~39 s, the second would end after ~75 s
        hint, _s = TRACK.project(sim.state.pos, hint)
        lateral = float(((sim.state.pos - TRACK.center[hint]) * TRACK.normal[hint]).sum())
        heading_error = obs[0, -1] * math.pi
        steer = np.clip(-3.0 * heading_error - 0.5 * lateral / (TRACK.width / 2), -1.0, 1.0)
        throttle = 0.4 if sim.state.speed[0] < 20.0 else 0.0
        r = sim.step(actions(1, steer=steer, throttle=throttle))
        obs = r.obs
        assert not r.terminated[0], "controller crashed or went the wrong way"
        if r.lap_completed[0]:
            lap_decisions.append(k)
            if len(lap_decisions) == 1:
                assert r.lap_time[0] == pytest.approx(sim.episode_time[0], rel=0.05)
                # The lap ends back at the spawn point (the start line), not at s = 0,
                # within one decision of driving (20 m/s * 4/60 s ~ 1.3 m).
                _, s_now = TRACK.project(sim.state.pos, hint)
                gap = (s_now[0] - START_S + TRACK.length / 2) % TRACK.length - TRACK.length / 2
                assert 0.0 <= gap < 2.0
        else:
            assert np.isnan(r.lap_time[0])
        if lap_decisions and k == lap_decisions[0] + 200:
            break  # drove well past the start line without a second lap
    assert len(lap_decisions) == 1
    assert sim.laps[0] == 1


@pytest.mark.parametrize("center", [0.0, START_S], ids=["s_wrap", "start_line"])
def test_start_line_oscillation_counts_no_lap(center):
    # Teleporting back and forth across a point must never count as a lap. center=0 is
    # where s wraps from `length` back to 0, the classic place for a lap-counting bug.
    sim = make_sim()
    sim.place_car(0, s=center)
    for k in range(60):
        pos, heading = TRACK.pose_at(np.array([center + (3.0 if k % 2 else -3.0)]))
        sim.state.pos[0] = pos[0]
        sim.state.heading[0] = heading[0]
        r = sim.step(np.zeros((1, 2)))
        assert not r.lap_completed[0]
        assert not r.terminated[0]
    assert sim.laps[0] == 0


def test_max_laps_truncates_with_lap_bonus():
    # Teleport the car forward 5 m per decision all the way round; with max_laps = 1 the
    # decision that completes the lap must pay the bonus and end the episode as truncated.
    cfg = dataclasses.replace(CFG, max_laps=1)
    sim = make_sim(cfg=cfg)
    for k in range(1, 200):
        pos, heading = TRACK.pose_at(np.array([START_S + 5.0 * k]))
        sim.state.pos[0] = pos[0]
        sim.state.heading[0] = heading[0]
        r = sim.step(np.zeros((1, 2)))
        if r.lap_completed[0]:
            break
    assert r.lap_completed[0] and r.truncated[0] and not r.terminated[0]
    assert 5.0 * k >= TRACK.length > 5.0 * (k - 1)
    assert r.reward[0] == pytest.approx(cfg.lap_bonus + cfg.progress_scale * 5.0 - cfg.time_cost,
                                        abs=0.05)
    # Laps are timed per physics substep: the teleport lands before substep 1 of decision k.
    assert r.lap_time[0] == pytest.approx(((k - 1) * cfg.action_repeat + 1) * cfg.dt)
    assert not sim.alive[0]


def test_wrong_way_terminates():
    sim = make_sim()
    sim.place_car(0, s=START_S, heading_offset=math.pi)
    # Time to drive wrong_way_m from rest at full throttle, ignoring drag.
    t_needed = math.sqrt(2 * CFG.wrong_way_m / CFG.car.max_accel)
    decision_s = CFG.dt * CFG.action_repeat
    limit = math.ceil(t_needed / decision_s) + 10
    for step in range(limit):
        r = sim.step(actions(1, throttle=1.0))
        if r.terminated[0]:
            break
    assert r.terminated[0], f"not terminated after {limit} decisions"
    assert r.reward[0] > -CFG.crash_penalty / 2  # wrong way, not a crash
    assert step + 1 >= math.floor(t_needed / decision_s)  # and not suspiciously early


def test_timeout_truncates():
    cfg = dataclasses.replace(CFG, max_episode_s=1.0)
    sim = make_sim(cfg=cfg)
    decisions = round(1.0 / (cfg.dt * cfg.action_repeat))  # 15
    for k in range(decisions):
        r = sim.step(np.zeros((1, 2)))
        assert r.truncated[0] == (k == decisions - 1)
        assert not r.terminated[0]
    assert sim.episode_time[0] == pytest.approx(1.0)


def test_auto_reset_returns_final_obs():
    sim = make_sim(n_cars=2, auto_reset=True)
    sim.place_car(0, s=10.0, lateral=TRACK.width / 2 - 0.5)
    r = sim.step(np.zeros((2, 2)))
    assert r.terminated[0]
    assert not np.allclose(r.final_obs[0], r.obs[0])
    assert sim.alive[0]
    # The respawned car starts fresh at the start line.
    assert sim.episode_time[0] == 0.0
    np.testing.assert_allclose(r.obs[0, CFG.n_rays:], 0.0, atol=1e-6)
    # Cars that are not done get identical obs and final_obs.
    assert not r.terminated[1]
    np.testing.assert_array_equal(r.final_obs[1], r.obs[1])


def test_frozen_cars_without_auto_reset():
    sim = make_sim(n_cars=2)
    sim.place_car(0, s=10.0, lateral=TRACK.width / 2 - 0.5)
    r = sim.step(actions(2, throttle=1.0))
    assert r.terminated[0]
    frozen_pos = sim.state.pos[0].copy()
    for _ in range(5):
        r = sim.step(actions(2, throttle=1.0))
        assert r.reward[0] == 0.0
        assert not sim.alive[0]
        assert not r.terminated[0]  # the end is reported once, on the decision it happens
        assert sim.alive[1] and r.reward[1] != 0.0
    np.testing.assert_array_equal(sim.state.pos[0], frozen_pos)
    sim.reset()
    assert sim.alive.all()


def test_seeded_random_start_reproducible():
    cfg = dataclasses.replace(CFG, random_start=True)
    a = BatchSim(TRACK, cfg, 8, seed=3).reset()
    b = BatchSim(TRACK, cfg, 8, seed=3).reset()
    c = BatchSim(TRACK, cfg, 8, seed=4).reset()
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert len(np.unique(a[:, 0])) > 1  # cars really start in different places


def test_place_car_sets_pose():
    sim = make_sim()
    sim.place_car(0, s=START_S + 20.0, lateral=2.0, speed=5.0)
    pos, heading = TRACK.pose_at(np.array([START_S + 20.0]))
    normal = np.array([-np.sin(heading[0]), np.cos(heading[0])])
    np.testing.assert_allclose(sim.state.pos[0], pos[0] + 2.0 * normal, atol=1e-9)
    assert sim.state.speed[0] == 5.0
    assert sim.laps[0] == 0 and sim.episode_time[0] == 0.0 and sim.alive[0]


def test_set_track_applies_on_reset():
    twisty = load_track("twisty")
    sim = make_sim()
    sim.set_track(twisty)
    assert sim.track is TRACK  # not yet
    sim.reset()
    assert sim.track is twisty
    pos, _ = twisty.pose_at(np.array([twisty.cum_len[twisty.start_index]]))
    np.testing.assert_allclose(sim.state.pos[0], pos[0], atol=1e-9)


def test_bench_reports_car_steps():
    from racer.bench import bench, format_table

    results = bench(n_cars_list=(1, 4), seconds=0.05)
    assert [r["n_cars"] for r in results] == [1, 4]
    for r in results:
        assert r["decisions"] > 0
        assert r["car_steps_per_s"] == pytest.approx(r["decisions"] * r["n_cars"] / r["seconds"])
    assert "car-steps/s" in format_table(results)
