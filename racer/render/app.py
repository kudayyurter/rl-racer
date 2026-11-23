"""The window loop shared by every viewer (drive now, watch later), with screenshot support.

Why this exists: drive and watch differ only in what they update and draw each frame. This
module owns everything else: opening the 1600x900 window, the 60 fps loop, the D key that
toggles the debug overlay, Esc to quit (raylib's default exit key), and the ``--screenshot``
mode that renders a fixed number of frames, saves a PNG and exits, so a viewer can be checked
without anyone watching it.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pyray as rl

SCREEN_W, SCREEN_H = 1600, 900
FPS = 60
BACKGROUND = (20, 22, 26, 255)


@dataclass
class AppState:
    """Flags any part of a viewer may read. ``overlay`` flips when D is pressed."""

    overlay: bool = False


# One shared instance: draw callbacks read app_state.overlay to decide what to draw.
app_state = AppState()


def run_app(update: Callable[[float], None], draw: Callable[[], None], title: str,
            screenshot: str | None = None, frames: int = 0, overlay: bool = False,
            setup: Callable[[], None] | None = None) -> None:
    """Open the window and run until Esc / window close (or until the screenshot is saved).

    ``update(dt)`` advances the viewer by ``dt`` seconds of real time; ``draw()`` draws one
    frame between raylib's begin/end drawing. ``setup()``, if given, runs once right after
    the window opens: GPU work such as baking the track texture needs the window's OpenGL
    context, so it cannot happen before ``run_app`` is called.

    With ``screenshot`` set, ``frames`` frames are rendered (at least one), the last one is
    saved as a PNG at that path, and the window closes. In that mode ``update`` gets exactly
    1/FPS per frame instead of the measured frame time, so screenshots are reproducible.
    """
    app_state.overlay = overlay
    rl.set_trace_log_level(rl.LOG_WARNING)   # raylib logs a lot at INFO level
    rl.set_config_flags(rl.FLAG_MSAA_4X_HINT)  # smooth edges on cars and overlay lines
    rl.init_window(SCREEN_W, SCREEN_H, title)
    try:
        rl.set_target_fps(FPS)
        # World-to-screen flips y, which mirrors polygon winding; never cull by winding.
        rl.rl_disable_backface_culling()
        if setup is not None:
            setup()
        frame = 0
        while not rl.window_should_close():
            if rl.is_key_pressed(rl.KEY_D):
                app_state.overlay = not app_state.overlay
            update(1.0 / FPS if screenshot else rl.get_frame_time())

            rl.begin_drawing()
            rl.clear_background(BACKGROUND)
            draw()
            frame += 1
            last = screenshot is not None and frame >= max(frames, 1)
            if last:
                _save_screen(screenshot)
            rl.end_drawing()
            if last:
                break
    finally:
        rl.close_window()


def _save_screen(path: str) -> None:
    """Save what has been drawn this frame as a PNG at ``path``.

    Called before ``end_drawing``: after it, the frame is swapped away and the buffer that
    is left is undefined. raylib batches draw calls, so the batch is flushed first. raylib's
    own ``take_screenshot`` writes relative to the working directory, so this resolves the
    path itself and exports the image directly.
    """
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    rl.rl_draw_render_batch_active()
    image = rl.load_image_from_screen()
    try:
        if not rl.export_image(image, str(out)):
            raise RuntimeError(f"could not write screenshot to {out}")
    finally:
        rl.unload_image(image)
    print(f"saved screenshot {out}")
