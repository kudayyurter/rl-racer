"""Debug overlay (what the agent "sees") and HUD text panels.

Why this exists: the overlay makes the simulator's hidden state visible. It draws the two walls
the crash check uses, the centerline progress is measured along, and the distance-sensor rays
of one car exactly as the observation reads them (``BatchSim.last_rays``). If a ray does not
end on a wall, the sensor is wrong; that is the fastest sanity check in the project. The HUD
helpers draw readable text boxes on top of everything, for drive and watch alike.
"""

import math

import numpy as np
import pyray as rl

from racer.render.camera import Vec2Array, world_to_screen
from racer.sim import BatchSim

WALL = (90, 220, 255, 255)        # cyan: stands out on asphalt, grass and kerbs
CENTERLINE = (255, 236, 120, 150)
RAY = (255, 70, 200, 255)          # magenta rays with a dot where each one hits
PANEL = (18, 20, 26, 200)
TEXT = (238, 238, 238, 255)
FONT_SIZE = 20
LINE_HEIGHT = 24
PADDING = 10
MARGIN = 12                       # gap between a panel and the screen edge
PANEL_ROW = FONT_SIZE + 2 * PADDING + 8   # height of a one-line panel plus a gap, for stacking

# The static overlay lines (walls, centerline) packed for raylib, cached because building
# them every frame would cost more than drawing them. Holds (key, track, arrays); the track
# is kept so its id() cannot be reused by another track while the entry is alive.
_line_cache: tuple | None = None


def format_lap_time(t: float | None) -> str:
    """Seconds as ``m:ss.mmm``; None or NaN (no lap yet) as ``-:--.---``."""
    if t is None or math.isnan(t):
        return "-:--.---"
    # Round to whole milliseconds first, so 59.9996 s becomes 1:00.000 and never 0:60.000.
    ms = int(round(t * 1000))
    minutes, rest = divmod(ms, 60_000)
    return f"{minutes}:{rest // 1000:02d}.{rest % 1000:03d}"


def draw_overlay(sim: BatchSim, camera, car: int = 0, *, panel_anchor: str,
                 panel_y_offset: float) -> None:
    """Walls, centerline, the sensor rays of car ``car`` and its progress around the lap.

    The progress readout is a ``draw_hud`` panel at ``panel_anchor`` / ``panel_y_offset``
    (see ``draw_hud``); each viewer picks a spot its own HUD leaves free.
    """
    track = sim.track
    left, right, center = _static_lines(track, camera)
    rl.draw_spline_linear(center.ptr, center.count, 1.5, CENTERLINE)
    rl.draw_spline_linear(left.ptr, left.count, 2.0, WALL)
    rl.draw_spline_linear(right.ptr, right.count, 2.0, WALL)

    # Rays: same fan BatchSim casts (evenly spread over ray_spread_deg around the heading).
    cfg = sim.cfg
    pos, heading = sim.state.pos[car], sim.state.heading[car]
    half = math.radians(cfg.ray_spread_deg) / 2
    angles = heading + np.linspace(-half, half, cfg.n_rays)
    dists = sim.last_rays[car]
    ends = pos + dists[:, None] * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    start = tuple(world_to_screen(camera, pos).tolist())
    for end, dist in zip(world_to_screen(camera, ends).tolist(), dists):
        rl.draw_line_ex(start, tuple(end), 1.5, RAY)
        if dist < cfg.ray_range:   # a ray at full range hit nothing, so it gets no dot
            rl.draw_circle_v(tuple(end), 3.5, RAY)

    # Progress from the start line. The nearest sample over the whole track is a fine hint
    # here: a car on the road is always far closer to its own stretch than to any other.
    nearest = int(np.argmin(((track.center - pos) ** 2).sum(axis=1)))
    _, s = track.project(pos[None], np.array([nearest]))
    done = (s[0] - track.cum_len[track.start_index]) % track.length
    text = (f"car {car}  progress {done:.1f} / {track.length:.1f} m "
            f"({100 * done / track.length:.1f}%)  laps {sim.laps[car]}")
    draw_hud([text], anchor=panel_anchor, y_offset=panel_y_offset)


def draw_hud(lines: list[str], anchor: str = "top-left", y_offset: float = 0.0) -> None:
    """Draw ``lines`` in a translucent panel in a screen corner.

    ``anchor`` is "top-left", "top-right", "bottom-left" or "bottom-right". ``y_offset``
    moves the panel that many pixels away from its top/bottom edge, to stack panels
    (``PANEL_ROW`` per one-line panel).
    """
    width = max((rl.measure_text(line, FONT_SIZE) for line in lines), default=0) + 2 * PADDING
    # The last line needs only FONT_SIZE, not a full LINE_HEIGHT, below its top.
    height = (len(lines) - 1) * LINE_HEIGHT + FONT_SIZE + 2 * PADDING
    x = MARGIN if anchor.endswith("left") else rl.get_screen_width() - MARGIN - width
    if anchor.startswith("top"):
        y = MARGIN + y_offset
    else:
        y = rl.get_screen_height() - MARGIN - height - y_offset
    x, y = int(x), int(y)   # draw_text takes whole pixels
    rl.draw_rectangle_rounded(rl.Rectangle(x, y, width, height), 0.15, 6, PANEL)
    for k, line in enumerate(lines):
        rl.draw_text(line, x + PADDING, y + PADDING + k * LINE_HEIGHT, FONT_SIZE, TEXT)


def _static_lines(track, camera) -> tuple[Vec2Array, Vec2Array, Vec2Array]:
    """Closed left wall, right wall and centerline in screen space, cached per track+camera."""
    global _line_cache
    offset, zoom = camera
    key = (id(track), tuple(np.asarray(offset).tolist()), float(zoom))
    if _line_cache is None or _line_cache[0] != key:
        arrays = tuple(Vec2Array(world_to_screen(camera, np.vstack([line, line[:1]])))
                       for line in (track.left, track.right, track.center))
        _line_cache = (key, track, arrays)
    return _line_cache[2]
