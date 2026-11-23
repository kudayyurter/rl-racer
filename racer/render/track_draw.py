"""Draw the "Clean circuit" track once into a texture, then show that texture every frame.

Why this exists: the track never changes while a viewer runs, but drawing it means hundreds of
road triangles, kerb blocks and stripes. Each raylib call from Python has real overhead, so
redrawing all that 60 times a second would eat the frame budget. Instead ``bake_track`` draws
everything once into an off-screen ``RenderTexture2D`` and ``TrackTexture.draw`` blits it with a
single call per frame.

The look, layer by layer: mowed-stripe grass, smooth asphalt between the two walls, red/white
kerbs just inside each wall, a white edge line a little inboard of each kerb, and a checkered
start line.
Everything drawn on top of the grass is drivable: the wall (where a car crashes) is the outer
edge of the kerb.

The texture is baked at ``SUPERSAMPLE`` times the screen resolution and shrunk with bilinear
filtering when drawn. Render textures get no multisampling, so this is what keeps curved
edges smooth instead of stair-stepped.
"""

import numpy as np
import pyray as rl

from racer.render.camera import Camera, Vec2Array, world_to_screen
from racer.track import Track

# --- Geometry, meters -------------------------------------------------------------------
KERB_WIDTH = 1.3        # kerb band just inside each wall
EDGE_LINE_WIDTH = 0.35  # white edge line inboard of each kerb...
EDGE_LINE_GAP = 0.45    # ...separated from it by a strip of asphalt, so the white line and
                        # the white kerb blocks don't merge into one band at full-track zoom
STRIPE_WIDTH = 12.0     # width of one mowed grass stripe
START_COLUMNS = 8       # checker squares across the road; the squares are width / 8 wide
START_ROWS = 2          # checker rows along the road
SUPERSAMPLE = 2         # bake resolution multiplier (2x2 samples per screen pixel)

# --- Palette (RGBA) -----------------------------------------------------------------------
GRASS_LIGHT = (112, 170, 84, 255)
GRASS_DARK = (100, 156, 75, 255)
ASPHALT = (62, 66, 72, 255)
EDGE_LINE = (236, 236, 230, 255)
KERB_RGBA = {"red": (206, 52, 48, 255), "white": (240, 240, 236, 255)}
CHECKER_DARK = (30, 32, 36, 255)
CHECKER_LIGHT = (244, 244, 240, 255)


def kerb_quads(track: Track, every: int = 2) -> np.ndarray:
    """Kerb blocks along both edges as world quads (Q, 4, 2): left edge first, then right.

    Each block spans about ``every`` centerline samples and runs from ``KERB_WIDTH`` inside
    the wall out to the wall. Corners are ordered: inner start, inner end, wall end, wall
    start, so consecutive blocks share an edge and the band has no gaps.

    The block count per edge is rounded down to an even number (block boundaries are spread
    evenly over the samples) so the red/white alternation also holds across the point where
    the loop closes; with an odd count two blocks of the same colour would touch there.
    """
    m = len(track.center)
    per_edge = max(2, (m // every) // 2 * 2)
    edges = np.round(np.linspace(0, m, per_edge + 1)).astype(int)
    start, end = edges[:-1], edges[1:] % m
    half = track.width / 2.0
    quads = []
    for side in (1.0, -1.0):  # +1: left wall (along +normal), -1: right wall
        wall = track.center + side * half * track.normal
        inner = track.center + side * (half - KERB_WIDTH) * track.normal
        quads.append(np.stack([inner[start], inner[end], wall[end], wall[start]], axis=1))
    return np.concatenate(quads)


def kerb_colors(q: int) -> list[str]:
    """Colour names for ``q`` kerb blocks: "red", "white", "red", ...

    Matches ``kerb_quads``: each edge has an even block count, so both edges start on red.
    """
    return ["red" if i % 2 == 0 else "white" for i in range(q)]


def start_line_quads(track: Track) -> tuple[np.ndarray, list[bool]]:
    """Checker squares across the road at the start sample: world quads (K, 4, 2) and a
    dark/light flag per square."""
    i = track.start_index
    c, t, n = track.center[i], track.tangent[i], track.normal[i]
    size = track.width / START_COLUMNS
    quads, dark = [], []
    for row in range(START_ROWS):
        # Rows are centered on the start sample, one square deep each.
        a0 = (row - START_ROWS / 2) * size
        for col in range(START_COLUMNS):
            b0 = -track.width / 2 + col * size
            corner = c + a0 * t + b0 * n
            quads.append([corner, corner + size * t, corner + size * t + size * n,
                          corner + size * n])
            dark.append((row + col) % 2 == 1)
    return np.array(quads), dark


class TrackTexture:
    """A baked track. ``draw()`` puts it on screen; ``unload()`` frees the GPU memory."""

    def __init__(self, target, screen_size: tuple[int, int]):
        self.target = target              # the RenderTexture2D holding the picture
        self.screen_size = screen_size    # (w, h) it covers on screen, in pixels

    def draw(self) -> None:
        tex = self.target.texture
        w, h = self.screen_size
        # Render textures are stored upside down (OpenGL's origin is bottom-left), so the
        # source rectangle uses a negative height to flip it back while drawing.
        source = rl.Rectangle(0, 0, tex.width, -tex.height)
        rl.draw_texture_pro(tex, source, rl.Rectangle(0, 0, w, h), rl.Vector2(0, 0), 0.0,
                            rl.WHITE)

    def unload(self) -> None:
        rl.unload_render_texture(self.target)


def bake_track(track: Track, camera: Camera, size: tuple[int, int] | None = None) -> TrackTexture:
    """Draw the whole static track once into a texture covering the screen (needs a window).

    ``size`` is the on-screen size in pixels; it defaults to the current window size.
    """
    w, h = size if size is not None else (rl.get_screen_width(), rl.get_screen_height())
    target = rl.load_render_texture(w * SUPERSAMPLE, h * SUPERSAMPLE)
    cam = camera.scaled(SUPERSAMPLE)   # same view, in texture pixels

    rl.begin_texture_mode(target)
    # The y flip mirrors every shape (see camera.py), so don't let raylib cull by winding.
    rl.rl_disable_backface_culling()
    _draw_grass(cam, w * SUPERSAMPLE, h * SUPERSAMPLE)
    half = track.width / 2.0
    _fill_band(track, cam, -half, half, ASPHALT)
    for side in (1.0, -1.0):
        line_out = half - KERB_WIDTH - EDGE_LINE_GAP
        _fill_band(track, cam, side * (line_out - EDGE_LINE_WIDTH), side * line_out, EDGE_LINE)
    quads = kerb_quads(track)
    _fill_quads(cam, quads, [KERB_RGBA[name] for name in kerb_colors(len(quads))])
    squares, dark = start_line_quads(track)
    _fill_quads(cam, squares, [CHECKER_DARK if d else CHECKER_LIGHT for d in dark])
    rl.end_texture_mode()

    # Bilinear filtering averages the supersampled pixels when the texture is drawn smaller.
    rl.set_texture_filter(target.texture, rl.TEXTURE_FILTER_BILINEAR)
    return TrackTexture(target, (w, h))


def _draw_grass(cam: Camera, tex_w: int, tex_h: int) -> None:
    """Fill the texture with alternating vertical stripes anchored to world x = 0.

    Anchoring to the world (not the screen) keeps the stripe pattern the same wherever the
    camera sits, so drive and watch views of a track look alike.
    """
    rl.clear_background(GRASS_LIGHT)
    stripe_px = STRIPE_WIDTH * cam.zoom
    # Index of the stripe under the left screen edge, from screen x = offset + zoom * world x.
    first = int(np.floor(-cam.offset[0] / stripe_px))
    k = first
    while (x := cam.offset[0] + k * stripe_px) < tex_w:
        if k % 2:
            rl.draw_rectangle_rec(rl.Rectangle(x, 0, stripe_px, tex_h), GRASS_DARK)
        k += 1


def _fill_band(track: Track, cam: Camera, d_from: float, d_to: float, color) -> None:
    """Fill the closed band between two offsets from the centerline (meters, + is left).

    Drawn as one triangle strip that zig-zags between the two offset curves and repeats
    the first pair at the end to close the loop.
    """
    a = track.center + d_from * track.normal
    b = track.center + d_to * track.normal
    zigzag = np.stack([a, b], axis=1).reshape(-1, 2)        # a0, b0, a1, b1, ...
    zigzag = np.vstack([zigzag, zigzag[:2]])                 # close the loop
    pts = Vec2Array(world_to_screen(cam, zigzag))
    rl.draw_triangle_strip(pts.ptr, pts.count, color)


def _fill_quads(cam: Camera, quads: np.ndarray, colors: list) -> None:
    """Fill world quads (Q, 4, 2), one colour each, as two triangles per quad."""
    screen = world_to_screen(cam, quads)
    for (p0, p1, p2, p3), color in zip(screen.tolist(), colors):
        rl.draw_triangle(p0, p1, p2, color)
        rl.draw_triangle(p0, p2, p3, color)
