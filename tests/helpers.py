"""Shared test helpers: small hand-made tracks and a wall self-intersection check."""

import numpy as np

from racer.geometry import any_crossing


def circle_points(radius: float, n: int) -> np.ndarray:
    """n control points on a circle, counter-clockwise (so the left wall is the inner one)."""
    a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([radius * np.cos(a), radius * np.sin(a)], axis=1)


def hairpin_points(gap: float, straight: float) -> np.ndarray:
    """A stadium: two parallel straights `gap` apart joined by semicircles (U-turns).

    Starts at the origin heading +x along the bottom straight, turns left (CCW).
    """
    r = gap / 2.0
    pts = [[x, 0.0] for x in np.linspace(0.0, straight, 7)[:-1]]
    for a in np.linspace(-np.pi / 2, np.pi / 2, 9)[:-1]:           # right-hand U-turn
        pts.append([straight + r * np.cos(a), r + r * np.sin(a)])
    pts += [[x, gap] for x in np.linspace(straight, 0.0, 7)[:-1]]
    for a in np.linspace(np.pi / 2, 3 * np.pi / 2, 9)[:-1]:        # left-hand U-turn
        pts.append([r * np.cos(a), r + r * np.sin(a)])
    return np.array(pts)


def wall_self_intersects(wall: np.ndarray) -> bool:
    """True if any two non-adjacent segments of the closed polyline `wall` (M,2) touch or cross.

    Adjacent segments share an endpoint, which any_crossing counts as touching, so each
    segment is tested only against segments that are not itself or its two neighbours.
    """
    m = len(wall)
    a, b = wall, np.roll(wall, -1, axis=0)
    for i in range(m):
        others = [j for j in range(m) if (j - i) % m not in (0, 1, m - 1)]
        if any_crossing(a[i][None, None], b[i][None, None],
                        a[others][None], b[others][None])[0]:
            return True
    return False


class ScriptedOvalPolicy:
    """A hand-written P-controller that laps the oval, wrapped as a ``Policy`` (obs -> action).

    It sees only the observation, like a learned policy. Steer against the heading error and
    the sideways offset. Ray 0 points right (negative angle) and the last ray points left, so
    (ray 0 - last ray) / 2 is how far the car sits left of the centerline. Throttle is 0.4 but
    coasts above 20 m/s, because the grip
    limit only allows about 24 m/s round the end turns.
    """

    def __init__(self, cfg, track):
        self.ray_range = cfg.ray_range
        self.half_width = track.width / 2
        self.max_speed = cfg.car.max_speed

    def act(self, obs: np.ndarray) -> np.ndarray:
        heading_error = obs[:, -1] * np.pi
        lateral = (obs[:, 0] - obs[:, -4]) / 2 * self.ray_range  # right ray - left ray
        steer = np.clip(-3.0 * heading_error - 0.5 * lateral / self.half_width, -1.0, 1.0)
        throttle = np.where(obs[:, -3] * self.max_speed < 20.0, 0.4, 0.0)
        return np.stack([steer, throttle], axis=1)

    def activations(self, obs):
        return [obs, self.act(obs)]

    def layer_weights(self):
        return []
