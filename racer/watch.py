"""Watch mode: see a training run learn, live (``racer watch RUN_DIR``).

Why this exists: training runs headless at full speed, writing checkpoints, metrics and ghost
laps into its run directory (see ``racer.runlog``). This viewer is a separate process that
only *reads* that directory, so watching never slows training down. Each time the trainer
saves a better policy, the car on screen starts driving with it: you watch the car learn lap
by lap.

What is on screen:

* left, the track: the current policy driving one car live, plus faded ghost cars replaying
  the best evaluation laps of earlier checkpoints (older ghosts are fainter). Ghosts run on
  the live car's lap clock, so they all leave the start line when it does and you can see
  who is ahead;
* right, a side panel: the policy network firing as it drives, learning curves from
  ``metrics.csv``, and a status readout (which checkpoint is driving, lap times).

Hot reload rules. Every ``reload_s`` seconds the viewer re-reads the metrics and ghosts and
checks the modification time of ``checkpoints/latest.pt``. A changed checkpoint is not swapped
in at once: it waits until the live car's episode ends, so a lap is never driven half by one
policy and half by another. If loading fails (for example a method whose loader does not
exist yet) the error is shown in the panel and the previous policy keeps driving.

The car runs the exact environment the run trained in: the env config comes from the run's
``config.json``, and so does the track unless ``--track`` picks another one.

Controls: D toggles the debug overlay (walls, centerline, sensor rays), Esc quits.
"""

import json
import math
from pathlib import Path

import pyray as rl
import torch

from racer.config import EnvConfig, from_dict
from racer.policy import Policy, policy_from_checkpoint
from racer.render.app import SCREEN_H, SCREEN_W, app_state, run_app
from racer.render.camera import fit_camera
from racer.render.car_draw import CAR_COLORS, draw_cars, draw_ghosts
from racer.render.curve import draw_curve
from racer.render.netpanel import draw_network
from racer.render.overlay import draw_hud, draw_overlay, format_lap_time
from racer.render.track_draw import bake_track
from racer.runlog import RunDir, latest_checkpoint
from racer.sim import BatchSim
from racer.track import Track, load_track

MAX_FRAME_S = 0.1   # longest real-time gap simulated in one frame (e.g. after dragging the window)

# --- Layout, pixels: track on the left, a fixed-width panel on the right ---------------
PANEL_W = 480
PANEL_X = SCREEN_W - PANEL_W
TRACK_RECT = (0, 0, PANEL_X, SCREEN_H)
PAD = 20                                         # inner margin of the panel
NET_RECT = (PANEL_X + PAD, 52, PANEL_W - 2 * PAD, 300)
CURVE_RECTS = [(PANEL_X + PAD, 374, PANEL_W - 2 * PAD, 112),
               (PANEL_X + PAD, 502, PANEL_W - 2 * PAD, 112)]
CURVE_KEYS = ["eval_best_lap_s", "eval_progress_m"]
STATUS_Y = 642
PANEL_BG = (28, 31, 38, 255)
PANEL_EDGE = (52, 57, 68, 255)
HEADING = (240, 242, 246, 255)
LABEL = (150, 156, 168, 255)
VALUE = (236, 238, 242, 255)
ERROR = (255, 120, 110, 255)
WAIT_BG = (18, 20, 26, 210)                      # behind the "waiting" message on the track
HEADING_SIZE = 20
TEXT_SIZE = 20
LINE_H = 24
VALUE_X = 120                                    # status values line up at this x offset
HELP = "D overlay   Esc quit"


class WatchSession:
    """The viewer's logic, with no drawing: polling the run, driving the car, hot reload.

    ``update(dt)`` is everything one frame does; the draw code only reads attributes.
    Keeping this free of raylib calls lets the tests drive it against a real run directory.
    """

    def __init__(self, run_path: str | Path, track_name: str | None = None,
                 reload_s: float = 3.0, n_ghosts: int = 5):
        self.run = RunDir.open(run_path)
        self.track_name = track_name
        self.reload_s = reload_s
        self.n_ghosts = n_ghosts

        # Filled once config.json can be read (it may not exist yet when we start).
        self.method: str | None = None
        self.track: Track | None = None
        self.cfg: EnvConfig | None = None
        self.sim: BatchSim | None = None
        self.obs = None

        self.policy: Policy | None = None
        self.checkpoint: dict | None = None   # the checkpoint the car is driving
        self.loaded_mtime: int | None = None  # latest.pt mtime when last loaded (or tried)
        self.pending = False                  # latest.pt changed; swap in at episode end
        self.error: str | None = None

        self.rows: list[dict] = []
        self.ghosts: list = []                # the newest n_ghosts, oldest first
        self._ghost_files: list[str] = []

        self.accumulator = 0.0     # real time not yet simulated, seconds
        self.lap_start = 0.0       # episode time when the current lap began
        self.last_lap = math.nan
        self.poll_left = reload_s
        self.poll()

    # ---------------------------------------------------------------- reading the run

    def poll(self) -> None:
        """Re-read metrics and ghosts, and note whether ``latest.pt`` has changed."""
        if self.sim is None:
            self._start()
        try:
            self.rows = self.run.read_metrics()
            files = sorted(p.name for p in (self.run.path / "ghosts").glob("ghost_*.npz"))
            if files != self._ghost_files:   # only re-load ghosts when there are new ones
                self.ghosts = self.run.ghosts()[-self.n_ghosts:] if self.n_ghosts else []
                self._ghost_files = files
        except (OSError, ValueError):
            pass   # caught the trainer mid-write; the next poll will see the finished file

        path = latest_checkpoint(self.run.path)
        if path is not None and path.stat().st_mtime_ns != self.loaded_mtime:
            self.pending = True
        # Nothing is driving yet, so there is no lap to protect: load straight away.
        if self.pending and self.policy is None and self.sim is not None:
            self._load()

    def _start(self) -> None:
        """Build the track and simulator from ``config.json``, if it is readable yet."""
        try:
            config = json.loads((self.run.path / "config.json").read_text())
        except (OSError, ValueError):
            return   # not written yet (or caught half written): try again next poll
        name = self.track_name or config.get("train", {}).get("track")
        if name is None:
            self.error = "config.json names no track: pass --track"
            return
        self.method = config.get("method")
        self.track = load_track(name)
        self.cfg = from_dict(EnvConfig, config.get("env", {}))
        self.sim = BatchSim(self.track, self.cfg, n_cars=1, auto_reset=False)
        self.obs = self.sim.reset()

    def _load(self) -> None:
        """Swap in ``latest.pt``; on failure keep the old policy and remember the error."""
        path = latest_checkpoint(self.run.path)
        self.pending = False
        if path is None:
            return
        # Remember this version even if it fails, so a broken file is tried once, not on
        # every poll; a newer save changes the mtime and gets its own try.
        self.loaded_mtime = path.stat().st_mtime_ns
        try:
            # weights_only=False: see racer.policy.load_policy (our own checkpoint format).
            ckpt = torch.load(path, weights_only=False)
            # A policy trained on other sensors cannot drive this env; say so plainly rather
            # than letting it surface as a numpy shape error from the first act().
            expected, given = ckpt.get("obs_dim"), self.obs.shape[1]
            if expected is not None and expected != given:
                self.error = (f"checkpoint expects {expected} observation inputs, this env "
                              f"gives {given} (config.json env mismatch?)")
                return
            policy = policy_from_checkpoint(ckpt, path)
            policy.act(self.obs)   # fail here, not mid-drive, if it can't read our observations
        except Exception as e:   # anything: the viewer must keep running
            self.error = f"could not load checkpoint: {type(e).__name__}: {e}"
            return
        self.policy, self.checkpoint, self.error = policy, ckpt, None

    # ---------------------------------------------------------------- driving

    @property
    def lap_clock(self) -> float:
        """Seconds since the live car's current lap began (the ghosts' clock too)."""
        if self.sim is None:
            return 0.0
        return float(self.sim.episode_time[0]) - self.lap_start

    def update(self, dt: float) -> None:
        """Advance by ``dt`` seconds of real time: poll when due, then drive in fixed steps."""
        self.poll_left -= dt
        if self.poll_left <= 0:
            self.poll_left = self.reload_s
            self.poll()
        if self.policy is None:
            return

        # Same fixed-step accumulator as drive mode: physics advances in whole decisions of
        # dt * action_repeat, whatever the frame rate.
        self.accumulator += min(dt, MAX_FRAME_S)
        step_dt = self.cfg.dt * self.cfg.action_repeat
        while self.accumulator >= step_dt:
            self.accumulator -= step_dt
            result = self.sim.step(self.policy.act(self.obs))
            self.obs = result.obs
            if result.lap_completed[0]:
                self.last_lap = float(result.lap_time[0])
                self.lap_start = float(self.sim.episode_time[0])
            if not self.sim.alive[0]:
                self._next_episode()

    def _next_episode(self) -> None:
        """The episode ended: take any newer checkpoint, then start again from the line."""
        if self.pending:
            self._load()
        self.obs = self.sim.reset()
        self.lap_start = 0.0

    # ---------------------------------------------------------------- what the panel says

    def status_lines(self) -> list[str]:
        """The status readout as "label<TAB>value" lines (value may be empty)."""
        lines = []
        if self.policy is None:
            lines.append("waiting for first checkpoint\t")
        else:
            ckpt = self.checkpoint
            lines.append(f"driving\t{_iteration(ckpt.get('iteration'), ckpt.get('env_steps'))}")
        if self.track is not None:
            lines.append(f"track\t{self.track.name}")
        if self.rows:
            last = self.rows[-1]
            lines.append(f"training\t{_iteration(last['iteration'], last['env_steps'])}")
        if self.pending and self.policy is not None:
            lines.append("queued\tnewer one, at episode end")
        if self.policy is not None:
            speed = float(self.sim.state.speed[0]) * 3.6
            lines.append(f"speed\t{speed:.0f} km/h")
            lines.append(f"lap\t{format_lap_time(self.lap_clock)}")
            lines.append(f"last lap\t{format_lap_time(self.last_lap)}")
        best = self._best_eval_lap()
        if best is not None:
            lines.append(f"best lap\t{format_lap_time(best[0])}  eval, iteration {best[1]}")
        if self.ghosts:
            first, last = self.ghosts[0].iteration, self.ghosts[-1].iteration
            which = f"iteration {first}" if first == last else f"iterations {first}-{last}"
            lines.append(f"ghosts\t{len(self.ghosts)}, {which}")
        if self.error:
            lines.append(f"error\t{self.error}")
        return lines

    def _best_eval_lap(self) -> tuple[float, int] | None:
        """Fastest evaluation lap logged so far and its iteration, or None before any."""
        laps = [(r["eval_best_lap_s"], int(r["iteration"])) for r in self.rows
                if not math.isnan(r.get("eval_best_lap_s", math.nan))]
        return min(laps) if laps else None


def _iteration(iteration, env_steps) -> str:
    text = f"iteration {int(iteration)}" if iteration is not None else "iteration ?"
    if env_steps is not None and not math.isnan(env_steps):
        text += f"  ({int(env_steps):,} steps)"
    return text


# -------------------------------------------------------------------- drawing

def _wrap(text: str, width: int, size: int) -> list[str]:
    """Split ``text`` into lines no wider than ``width`` pixels (at least one word each)."""
    lines, line = [], ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if line and rl.measure_text(candidate, size) > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    return lines + [line] if line else lines


def _draw_panel(session: WatchSession) -> None:
    """The right-hand panel: network on top, learning curves, then the status readout."""
    rl.draw_rectangle(PANEL_X, 0, PANEL_W, SCREEN_H, PANEL_BG)
    rl.draw_line_ex((PANEL_X, 0), (PANEL_X, SCREEN_H), 2.0, PANEL_EDGE)
    x = PANEL_X + PAD

    title = "policy network"
    if session.method:
        title += f"  ({session.method})"
    rl.draw_text(title, x, 16, HEADING_SIZE, HEADING)
    if session.policy is not None:
        draw_network(session.policy.activations(session.obs), session.policy.layer_weights(),
                     NET_RECT)
    else:
        rl.draw_text("no checkpoint yet", x, NET_RECT[1] + 10, TEXT_SIZE, LABEL)

    for key, rect in zip(CURVE_KEYS, CURVE_RECTS):
        draw_curve(session.rows, key, rect)

    y = STATUS_Y
    rl.draw_line_ex((x, y - 10), (SCREEN_W - PAD, y - 10), 1.0, PANEL_EDGE)
    # Each status line becomes rows of (label, value text): long values wrap onto extra rows.
    rows = []
    for line in session.status_lines():
        label, value = line.split("\t", 1)
        if not value:   # a heading-style line, e.g. "waiting for first checkpoint"
            rows.append((label, "", HEADING))
            continue
        color = ERROR if label == "error" else VALUE
        for k, part in enumerate(_wrap(value, PANEL_W - PAD - VALUE_X - PAD, TEXT_SIZE)):
            rows.append((label if k == 0 else "", part, color))
    for label, value, color in rows:
        if y > SCREEN_H - LINE_H:
            break   # out of room: better cut off than draw past the window edge
        rl.draw_text(label, x, y, TEXT_SIZE, HEADING if not value else LABEL)
        rl.draw_text(value, x + VALUE_X, y, TEXT_SIZE, color)
        y += LINE_H


def _draw_waiting(session: WatchSession) -> None:
    """Centered message over the track area while there is nothing to drive."""
    lines = ["waiting for first checkpoint", session.run.path.name]
    cx, cy = PANEL_X / 2, SCREEN_H / 2
    box_w = rl.measure_text(lines[0], 40) + 80
    rl.draw_rectangle_rounded(rl.Rectangle(cx - box_w / 2, cy - 64, box_w, 128), 0.2, 8,
                              WAIT_BG)
    for k, (text, size) in enumerate(zip(lines, (40, 20))):
        w = rl.measure_text(text, size)
        rl.draw_text(text, int(cx - w / 2), int(cy - 40 + k * 56), size, VALUE)


def run_watch(run_path: str, track_name: str | None = None, reload_s: float = 3.0,
              n_ghosts: int = 5, screenshot: str | None = None, frames: int = 0) -> int:
    """Open the watch window on run directory ``run_path``."""
    if not Path(run_path).is_dir():
        print(f"racer watch: no run directory at {run_path}")
        return 2
    session = WatchSession(run_path, track_name, reload_s=reload_s, n_ghosts=n_ghosts)
    view = {}   # camera and baked track, made once the track is known and a window exists

    def ensure_track() -> None:
        # The track can become known after the window opens (config.json written late).
        if session.track is not None and "texture" not in view:
            camera = fit_camera(session.track.bounds(), SCREEN_W, SCREEN_H, rect=TRACK_RECT)
            view["camera"] = camera
            view["texture"] = bake_track(session.track, camera, size=(PANEL_X, SCREEN_H))

    def update(dt: float) -> None:
        session.update(dt)
        ensure_track()

    def draw() -> None:
        if "texture" in view:
            view["texture"].draw()
        if session.policy is not None:
            camera = view["camera"]
            draw_ghosts(session.ghosts, session.lap_clock, session.cfg.car, camera)
            draw_cars(session.sim.state, session.cfg.car, camera, CAR_COLORS[:1])
            if app_state.overlay:
                draw_overlay(session.sim, camera, car=0, panel_anchor="bottom-left",
                             panel_y_offset=0)
        else:
            _draw_waiting(session)
        draw_hud([HELP], anchor="top-left")
        _draw_panel(session)

    def teardown() -> None:
        if "texture" in view:
            view["texture"].unload()

    run_app(update, draw, f"RL Racer - watch ({Path(run_path).name})", screenshot=screenshot,
            frames=frames, setup=ensure_track, teardown=teardown)
    return 0
