"""Draw cars: a rounded, coloured body with a darker windshield, over a soft drop shadow.

Why this exists: drive mode, the watch viewer and its ghost replays all draw cars the same way.
Shapes are built once in the car's own frame (x forward, y left, meters) and then rotated and
moved per car, using the same heading convention as ``racer.car``. The body is the physics
rectangle (``CarParams.length`` x ``width``) with rounded corners, so what you see is what
the crash check uses, give or take the corner rounding.
"""

import numpy as np
import pyray as rl

from racer.car import CarState
from racer.config import CarParams
from racer.render.camera import world_to_screen

# One colour per car; drive uses the first. Picked to stand out on grey asphalt and green
# grass, and to stay distinct from the red/white kerbs.
CAR_COLORS = [
    (250, 192, 28),    # yellow
    (52, 152, 235),    # blue
    (236, 84, 160),    # pink
    (64, 200, 140),    # teal
    (245, 124, 40),    # orange
    (160, 120, 240),   # violet
]
GLASS = (28, 36, 50)          # windshield and rear window
SHADOW = (0, 0, 0, 90)        # translucent black, so the ground shows through
SHADOW_OFFSET_M = 0.7         # shadow shift down-right on screen, meters (scales with zoom)
CORNER_RADIUS_M = 0.55        # body corner rounding
CORNER_POINTS = 4             # points per rounded corner


def _rounded_rect(length: float, width: float, radius: float) -> np.ndarray:
    """Outline (K, 2) of a rounded rectangle centered on the origin, counter-clockwise."""
    hl, hw = length / 2 - radius, width / 2 - radius
    centers = [(hl, hw), (-hl, hw), (-hl, -hw), (hl, -hw)]   # FL, RL, RR, FR (CCW)
    pts = []
    for k, (cx, cy) in enumerate(centers):
        # Each corner sweeps a quarter turn, starting where the previous one ended.
        angles = np.linspace(0, np.pi / 2, CORNER_POINTS) + k * np.pi / 2
        pts.append(np.stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)], axis=1))
    return np.concatenate(pts)


def _windows(length: float, width: float) -> list[np.ndarray]:
    """Windshield and rear window as local quads, sized relative to the body."""
    hl, hw = length / 2, width / 2
    # Top-down, glass is a trapezoid: wide where it meets the bodywork, narrow at the roof.
    windshield = np.array([[0.05 * hl, 0.64 * hw], [0.05 * hl, -0.64 * hw],
                           [0.42 * hl, -0.80 * hw], [0.42 * hl, 0.80 * hw]])
    rear = np.array([[-0.48 * hl, 0.62 * hw], [-0.48 * hl, -0.62 * hw],
                     [-0.62 * hl, -0.74 * hw], [-0.62 * hl, 0.74 * hw]])
    return [windshield, rear]


def _place(local: np.ndarray, pos: np.ndarray, heading: float) -> np.ndarray:
    """Rotate local points (K, 2) by ``heading`` and move them to ``pos`` (world meters)."""
    c, s = np.cos(heading), np.sin(heading)
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + pos


def _rgba(color, alpha: float) -> tuple[int, int, int, int]:
    """(r, g, b) or (r, g, b, a) scaled by ``alpha`` in [0, 1] -> an RGBA tuple for raylib."""
    a = color[3] if len(color) == 4 else 255
    return (color[0], color[1], color[2], int(round(a * alpha)))


def _darker(color, factor: float = 0.55) -> tuple[int, int, int]:
    return tuple(int(c * factor) for c in color[:3])


def draw_cars(states: CarState, p: CarParams, camera, colors: list, alpha: float = 1.0) -> None:
    """Draw every car in ``states``; car ``i`` uses ``colors[i % len(colors)]``.

    ``alpha`` fades whole cars (shadow included), for ghost replays. Shadows are drawn first
    for all cars, so a shadow never lands on top of a neighbouring car.
    """
    _, zoom = camera
    body = _rounded_rect(p.length, p.width, min(CORNER_RADIUS_M, p.width / 2))
    windows = _windows(p.length, p.width)
    shadow_shift = np.array([1.0, 1.0]) * SHADOW_OFFSET_M * zoom   # screen pixels, down-right

    outlines = [world_to_screen(camera, _place(body, states.pos[i], states.heading[i]))
                for i in range(len(states.pos))]
    shadow = _rgba(SHADOW, alpha)
    for outline in outlines:
        _fill_convex(outline + shadow_shift, shadow)

    for i, outline in enumerate(outlines):
        color = colors[i % len(colors)]
        _fill_convex(outline, _rgba(color, alpha))
        glass = _rgba(GLASS, alpha)
        for window in windows:
            pts = world_to_screen(camera, _place(window, states.pos[i], states.heading[i]))
            _fill_convex(pts, glass)
        # A thin darker rim keeps the body crisp against both asphalt and grass.
        rim = np.vstack([outline, outline[:1]])
        rl.draw_line_strip([tuple(q) for q in rim.tolist()], len(rim), _rgba(_darker(color), alpha))


def _fill_convex(screen_pts: np.ndarray, color) -> None:
    """Fill a convex polygon given as screen points (K, 2), as a triangle fan."""
    pts = [tuple(q) for q in screen_pts.tolist()]
    rl.draw_triangle_fan(pts, len(pts), color)
