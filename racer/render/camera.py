"""Camera: the one mapping from world meters to screen pixels, shared by every draw call.

Why this exists: the simulator works in meters with y pointing up (math convention), while
raylib draws in pixels with y pointing down. Keeping the conversion in one function means the
track, the cars, the overlay and the watch viewer can never disagree about where things are.

Convention (used everywhere)::

    screen = offset + zoom * (x, -y)

``offset`` is the screen position of the world origin, in pixels; ``zoom`` is pixels per meter.
The minus sign on y is the flip from y-up to y-down. A side effect worth knowing: the flip
mirrors shapes, so a polygon that winds counter-clockwise in the world winds clockwise on
screen. The renderer turns raylib's back-face culling off so winding never matters.
"""

from typing import NamedTuple

import numpy as np
import pyray as rl


class Camera(NamedTuple):
    """A fixed 2D camera. It is a plain tuple, so ``offset, zoom = camera`` works."""

    offset: np.ndarray  # (2,) screen pixel position of world (0, 0)
    zoom: float         # pixels per meter

    def scaled(self, factor: float) -> "Camera":
        """The same view rendered at ``factor`` times the resolution (used for supersampling)."""
        return Camera(self.offset * factor, self.zoom * factor)


def fit_camera(bounds, screen_w: int, screen_h: int, margin: float = 0.05,
               rect: tuple[float, float, float, float] | None = None) -> Camera:
    """Fit world ``bounds`` (xmin, ymin, xmax, ymax) inside a screen rectangle, centered.

    ``rect`` is the target area as (x, y, w, h) in pixels; it defaults to the whole screen.
    The watch viewer passes a smaller rect to leave room for side panels. ``margin`` is the
    fraction of the rect left empty on each side. The aspect ratio is kept (one zoom for
    both axes), so the bounds fill the rect in one direction and are centered in the other.
    """
    x, y, w, h = rect if rect is not None else (0.0, 0.0, float(screen_w), float(screen_h))
    xmin, ymin, xmax, ymax = bounds
    zoom = min(w * (1 - 2 * margin) / (xmax - xmin), h * (1 - 2 * margin) / (ymax - ymin))
    # Put the world center of the bounds at the rect center. Solving
    # rect_center = offset + zoom * (cx, -cy) for offset gives:
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    offset = np.array([x + w / 2 - zoom * cx, y + h / 2 + zoom * cy])
    return Camera(offset, float(zoom))


def world_to_screen(camera, points: np.ndarray) -> np.ndarray:
    """Convert world points (..., 2) in meters to screen points (..., 2) in pixels."""
    offset, zoom = camera
    pts = np.asarray(points, dtype=float)
    return np.asarray(offset) + zoom * pts * np.array([1.0, -1.0])


class Vec2Array:
    """Screen points packed as a C ``Vector2`` array, for raylib calls that take many points.

    Building the array straight from numpy memory avoids creating one Python object per
    point, which matters when the overlay passes hundreds of wall points every frame.
    Keep the object alive while raylib uses ``ptr`` (it borrows the numpy buffer).
    """

    def __init__(self, screen_points: np.ndarray):
        self._data = np.ascontiguousarray(screen_points, dtype=np.float32).reshape(-1, 2)
        self.count = len(self._data)
        # pyray would take the address of an array a second time, so hand it a plain pointer.
        self.ptr = rl.ffi.cast("Vector2 *", rl.ffi.from_buffer("Vector2[]", self._data))
