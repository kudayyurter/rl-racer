"""Pure (window-free) tests for the render helpers: camera fitting, kerb geometry, time format.

None of these open a window; they only check the numpy geometry the renderer draws.
"""

import numpy as np
import pytest

from racer.render.camera import fit_camera, world_to_screen
from racer.render.overlay import format_lap_time
from racer.render.track_draw import KERB_WIDTH, kerb_colors, kerb_quads
from racer.track import load_track


def _corners(bounds):
    xmin, ymin, xmax, ymax = bounds
    return np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])


@pytest.mark.parametrize("name", ["oval", "twisty"])
@pytest.mark.parametrize("size", [(1600, 900), (900, 1600), (640, 480)])
def test_fit_camera_contains_bounds(name, size):
    w, h = size
    bounds = load_track(name).bounds()
    cam = fit_camera(bounds, w, h)
    pts = world_to_screen(cam, _corners(bounds))
    assert np.all(pts[:, 0] >= 0) and np.all(pts[:, 0] <= w)
    assert np.all(pts[:, 1] >= 0) and np.all(pts[:, 1] <= h)
    # It should actually fill one screen dimension (minus the margin), not just fit.
    span = pts.max(axis=0) - pts.min(axis=0)
    assert span[0] == pytest.approx(0.9 * w) or span[1] == pytest.approx(0.9 * h)


def test_fit_camera_flips_y_and_respects_target_rect():
    bounds = (0.0, 0.0, 100.0, 50.0)
    rect = (400, 100, 800, 600)
    offset, zoom = fit_camera(bounds, 1600, 900, rect=rect)
    pts = world_to_screen((offset, zoom), _corners(bounds))
    # World y-up becomes screen y-down: the top edge (y=50) is drawn above the bottom (y=0).
    assert pts[2, 1] < pts[0, 1]
    x, y, w, h = rect
    assert np.all((pts[:, 0] >= x) & (pts[:, 0] <= x + w))
    assert np.all((pts[:, 1] >= y) & (pts[:, 1] <= y + h))
    # Centered in the rect.
    assert pts[:, 0].mean() == pytest.approx(x + w / 2)
    assert pts[:, 1].mean() == pytest.approx(y + h / 2)


@pytest.mark.parametrize("every", [1, 2, 3])
def test_kerb_quads_alternate(every):
    track = load_track("twisty")
    m = len(track.center)
    quads = kerb_quads(track, every=every)
    q = len(quads)
    assert quads.shape == (q, 4, 2)

    # Both edges get the same number of kerb blocks, an even number per edge so the colours
    # still alternate where the loop closes, and about one block per `every` samples.
    per_edge = q // 2
    assert q == 2 * per_edge and per_edge % 2 == 0
    assert abs(per_edge - m / every) <= 2

    colors = kerb_colors(q)
    assert len(colors) == q and set(colors) == {"red", "white"}
    for edge in (colors[:per_edge], colors[per_edge:]):
        assert all(a != b for a, b in zip(edge, edge[1:]))
        assert edge[0] != edge[-1]  # alternation holds across the start/finish seam

    # Every corner lies on the road edge, at most width/2 + kerb width from the centerline.
    corners = quads.reshape(-1, 2)
    dist = np.linalg.norm(corners[:, None, :] - track.center[None, :, :], axis=2).min(axis=1)
    assert np.all(dist <= track.width / 2 + KERB_WIDTH + 1e-9)
    assert np.all(dist >= track.width / 2 - KERB_WIDTH - 1e-6)


def test_kerb_quads_cover_the_whole_loop_without_gaps():
    track = load_track("oval")
    quads = kerb_quads(track, every=2)
    per_edge = len(quads) // 2
    for edge in (quads[:per_edge], quads[per_edge:]):
        # Each block starts where the previous one ended (corners: inner start, inner end,
        # outer end, outer start), including the wrap from the last block to the first.
        nxt = np.roll(edge, -1, axis=0)
        np.testing.assert_allclose(edge[:, 1], nxt[:, 0])
        np.testing.assert_allclose(edge[:, 2], nxt[:, 3])


@pytest.mark.parametrize("t, text", [
    (0.0, "0:00.000"),
    (61.2341, "1:01.234"),
    (59.9996, "1:00.000"),   # rounds to the millisecond before splitting minutes
    (None, "-:--.---"),
    (float("nan"), "-:--.---"),
])
def test_format_lap_time(t, text):
    assert format_lap_time(t) == text


def _session(**overrides):
    import dataclasses

    from racer.config import EnvConfig
    from racer.drive import DriveSession
    from racer.sim import BatchSim

    cfg = dataclasses.replace(EnvConfig(), action_repeat=1, **overrides)
    return DriveSession(BatchSim(load_track("oval"), cfg, n_cars=1, auto_reset=True))


def test_drive_session_steps_physics_in_fixed_steps():
    session = _session()
    # ~0.5 s of frames at an uneven rate still simulates exactly 30 physics steps of 1/60 s
    # (the 0.001 extra keeps float rounding from leaving the 30th step in the accumulator).
    for dt in [0.01, 0.03, 0.02, 0.04, 0.025, 0.05, 0.075, 0.1, 0.1, 0.051]:
        session.update(dt, np.array([0.0, 1.0]))
    assert session.sim.episode_time[0] == pytest.approx(30 / 60)
    assert session.sim.state.speed[0] > 0
    assert session.hud_lines()[1] == "lap   0:00.500"


def test_drive_session_caps_long_frames():
    session = _session()
    session.update(5.0, np.array([0.0, 1.0]))   # e.g. the window was dragged for 5 s
    assert session.sim.episode_time[0] <= 0.1 + 1e-9


def test_drive_session_crash_respawns_and_resets_lap_clock():
    session = _session()
    for _ in range(600):   # full throttle, steering hard left: hits the wall within 10 s
        session.update(1 / 60, np.array([1.0, 1.0]))
        if session.message_left > 0:
            break
    assert session.message_left > 0
    assert session.lap_start == 0.0
    assert session.sim.episode_time[0] < 1.0   # respawned, clock restarted


@pytest.mark.parametrize("keys, action", [
    ({"KEY_UP"}, [0.0, 1.0]),
    ({"KEY_W", "KEY_A"}, [1.0, 1.0]),        # steer +1 is left (counter-clockwise)
    ({"KEY_S", "KEY_RIGHT"}, [-1.0, -1.0]),
    ({"KEY_D"}, [0.0, 0.0]),                 # D toggles the overlay, it does not steer
    ({"KEY_LEFT", "KEY_RIGHT"}, [0.0, 0.0]),
])
def test_keyboard_action_mapping(monkeypatch, keys, action):
    import racer.drive as drive

    pressed = {getattr(drive.rl, name) for name in keys}
    monkeypatch.setattr(drive.rl, "is_key_down", lambda k: k in pressed)
    np.testing.assert_array_equal(drive.keyboard_action(), action)
