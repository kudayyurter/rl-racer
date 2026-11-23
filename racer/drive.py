"""Drive mode: race the track yourself with the keyboard (``racer drive``).

Why this exists: before training anything, it helps to feel the car. Driving shows how much
the grip limit forces you to brake for corners and gives a human lap time to compare agents
against. It also exercises the whole render stack (camera, baked track, cars, overlay, HUD)
that the watch viewer reuses.

The car is car 0 of the real ``BatchSim``, so it obeys exactly the physics, crash and lap
rules the agents train under. Two settings change, neither of which affects the physics:
``action_repeat`` is 1, so the keyboard is read every physics step instead of every fourth;
and the episode never ends by time or lap count, so you can keep lapping. Crashing or driving
the wrong way still ends the episode, and auto-reset puts you back on the start line.

Controls: Up/W throttle, Down/S brake, Left/A steer left, Right steer right, R restart,
D toggle overlay (that is why D is not "steer right"), Esc quit.
"""

import dataclasses
import math
from pathlib import Path

import numpy as np
import pyray as rl

from racer.config import load_env_config
from racer.render.app import SCREEN_H, SCREEN_W, app_state, run_app
from racer.render.camera import fit_camera
from racer.render.car_draw import CAR_COLORS, draw_cars
from racer.render.overlay import draw_hud, draw_overlay, format_lap_time
from racer.render.track_draw import bake_track
from racer.sim import BatchSim
from racer.track import load_track

DEFAULT_ENV_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "env.toml"
HELP = "Up/W gas  Down/S brake  Left/A Right steer  R restart  D overlay  Esc quit"
MESSAGE_S = 2.0    # how long "crashed" stays on the HUD
MAX_FRAME_S = 0.1  # longest real-time gap simulated in one frame (e.g. after dragging the window)


def keyboard_action() -> np.ndarray:
    """Read the keys into one ``[steer, throttle]`` action. Steer +1 turns left (CCW)."""
    def down(*keys) -> bool:
        return any(rl.is_key_down(k) for k in keys)

    steer = float(down(rl.KEY_LEFT, rl.KEY_A)) - float(down(rl.KEY_RIGHT))
    throttle = float(down(rl.KEY_UP, rl.KEY_W)) - float(down(rl.KEY_DOWN, rl.KEY_S))
    return np.array([steer, throttle])


class DriveSession:
    """One car on one track, stepped in fixed physics steps, plus lap-time bookkeeping."""

    def __init__(self, sim: BatchSim):
        self.sim = sim
        self.accumulator = 0.0     # real time not yet simulated, seconds
        self.lap_start = 0.0       # episode time when the current lap began
        self.last_lap = float("nan")
        self.best_lap = float("nan")
        self.message, self.message_left = "", 0.0
        sim.reset()

    def restart(self) -> None:
        """Back to the start line. The session's best lap is kept."""
        self.sim.reset()
        self.lap_start = 0.0
        self.accumulator = 0.0

    def update(self, dt: float, action: np.ndarray) -> None:
        """Advance by ``dt`` seconds of real time with ``action`` held throughout.

        Physics always steps at the config's fixed ``dt`` (1/60 s); real frame time is
        collected in an accumulator and spent in whole steps. That keeps the car's behaviour
        identical to training no matter how fast the screen refreshes.
        """
        self.message_left = max(0.0, self.message_left - dt)
        self.accumulator += min(dt, MAX_FRAME_S)
        step_dt = self.sim.cfg.dt * self.sim.cfg.action_repeat
        while self.accumulator >= step_dt:
            self.accumulator -= step_dt
            result = self.sim.step(action[None, :])
            if result.lap_completed[0]:
                self.last_lap = float(result.lap_time[0])
                if math.isnan(self.best_lap) or self.last_lap < self.best_lap:
                    self.best_lap = self.last_lap
                self.lap_start = float(self.sim.episode_time[0])
            if result.terminated[0] or result.truncated[0]:
                # Auto-reset already respawned the car, so its episode clock is back at 0.
                self.lap_start = 0.0
                self.message, self.message_left = "Crashed (or wrong way): back to the start", MESSAGE_S

    def hud_lines(self) -> list[str]:
        speed_kmh = self.sim.state.speed[0] * 3.6
        current = float(self.sim.episode_time[0]) - self.lap_start
        lines = [
            f"{speed_kmh:5.0f} km/h",
            f"lap   {format_lap_time(current)}",
            f"last  {format_lap_time(self.last_lap)}",
            f"best  {format_lap_time(self.best_lap)}",
        ]
        if self.message_left > 0:
            lines.append(self.message)
        return lines


def run_drive(track_name: str = "twisty", overlay: bool = False, screenshot: str | None = None,
              frames: int = 0) -> int:
    """Open the drive window. ``track_name`` is a track name or a path to a track JSON."""
    track = load_track(track_name)
    cfg = load_env_config(DEFAULT_ENV_CONFIG)
    # See the module docstring: read input every physics step, and never end on time or laps.
    cfg = dataclasses.replace(cfg, action_repeat=1, max_episode_s=24 * 3600.0, max_laps=10**9)
    session = DriveSession(BatchSim(track, cfg, n_cars=1, auto_reset=True))
    camera = fit_camera(track.bounds(), SCREEN_W, SCREEN_H)
    baked = {}   # filled by setup() once the window (and its GPU context) exists

    def setup() -> None:
        baked["track"] = bake_track(track, camera)

    def update(dt: float) -> None:
        if rl.is_key_pressed(rl.KEY_R):
            session.restart()
        session.update(dt, keyboard_action())

    def draw() -> None:
        baked["track"].draw()
        draw_cars(session.sim.state, cfg.car, camera, CAR_COLORS[:1])
        if app_state.overlay:
            draw_overlay(session.sim, camera, car=0)
        draw_hud(session.hud_lines())
        draw_hud([HELP], anchor="top-right")

    run_app(update, draw, f"RL Racer - drive ({track.name})", screenshot=screenshot,
            frames=frames, overlay=overlay, setup=setup)
    return 0
