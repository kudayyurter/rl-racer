"""Draw the policy network live: what each neuron is doing right now, and how they connect.

Why this exists: a trained policy is just a few weight matrices, which says nothing on its
own. Drawing it while it drives shows the network reacting: a ray shortening ahead of a corner
lights up its input node, the hidden layer shifts, and the steer output swings. That makes
"the network maps sensor readings to actions" something you can see.

Layout: one column per layer, input on the left, output on the right, nodes spread top to
bottom. Node colour is the activation: blue for negative, orange for positive, grey near zero.
Edge colour is the weight's sign (same blue/orange) and its opacity grows with |weight|. Only
the strongest ``EDGES_PER_LAYER`` edges of each layer are drawn: a 12x32 layer already has 384
edges, and drawing all of them costs frame time and hides the important ones in a grey haze.
"""

import numpy as np
import pyray as rl

NEGATIVE = (70, 150, 255)       # blue
POSITIVE = (255, 150, 50)       # orange
NEUTRAL = (92, 98, 110)         # an activation near zero
NODE_RIM = (16, 18, 22, 255)
LABEL = (200, 204, 212, 255)
LABEL_SIZE = 20                 # raylib's built-in font is sharpest at multiples of 10 px
EDGES_PER_LAYER = 90
MAX_NODE_GAP = 34.0             # small layers are not stretched over the whole height
EDGE_ALPHA_MAX = 200
OUTPUT_LABELS = ["steer", "throttle"]


def input_labels(n_inputs: int) -> list[str]:
    """Names of the observation entries: the rays (right to left), then speed, steer, heading.

    Matches ``BatchSim._observe``: ``n_rays`` ray distances followed by three car readings.
    """
    return [f"ray {k}" for k in range(1, n_inputs - 2)] + ["speed", "steer", "heading"]


def node_layout(layer_sizes: list[int], rect: tuple[float, float, float, float]) -> list[np.ndarray]:
    """Screen positions of every node, one (n_i, 2) array per layer, all inside ``rect``.

    ``rect`` is (x, y, w, h) in pixels. The first layer sits on the left edge and the last on
    the right edge, with the others evenly between. Within a layer the nodes are evenly spaced
    and centered vertically; the gap is capped at ``MAX_NODE_GAP`` so that the two output
    nodes sit near each other instead of at opposite ends of the panel.
    """
    x, y, w, h = rect
    n_layers = len(layer_sizes)
    layout = []
    for k, n in enumerate(layer_sizes):
        col_x = x + w / 2 if n_layers == 1 else x + w * k / (n_layers - 1)
        gap = min(h / n, MAX_NODE_GAP)
        # Node j sits at the middle of its own gap-tall slot; the slots are centered in rect.
        top = y + (h - gap * n) / 2
        ys = top + gap * (np.arange(n) + 0.5)
        layout.append(np.stack([np.full(n, col_x), ys], axis=1))
    return layout


def top_edges(weights: np.ndarray, k: int) -> list[tuple[int, int, float]]:
    """The ``k`` largest-|w| connections of one layer as (input i, output j, weight).

    ``weights`` is (n_in, n_out). The result is ordered strongest first.
    """
    flat = np.abs(weights).ravel()
    k = min(k, flat.size)
    # argpartition finds the k largest without sorting all n_in * n_out of them.
    best = np.argpartition(flat, flat.size - k)[flat.size - k:]
    best = best[np.argsort(flat[best])[::-1]]
    rows, cols = np.unravel_index(best, weights.shape)
    return [(int(i), int(j), float(weights[i, j])) for i, j in zip(rows, cols)]


def _activation_color(a: float) -> tuple[int, int, int, int]:
    """Blend from grey at 0 to full blue at -1 or full orange at +1."""
    a = float(np.clip(a, -1.0, 1.0))
    target = POSITIVE if a > 0 else NEGATIVE
    f = abs(a)
    return tuple(int(n + f * (t - n)) for n, t in zip(NEUTRAL, target)) + (255,)


def draw_network(activations: list[np.ndarray], weights: list[np.ndarray],
                 rect: tuple[float, float, float, float]) -> None:
    """Draw the network inside ``rect`` (x, y, w, h), labels included.

    ``activations`` is ``Policy.activations(obs)`` for one observation: the input, each
    hidden layer, then the output. ``weights`` is ``Policy.layer_weights()``, one (in, out)
    matrix per layer; a matrix whose shape does not fit the activations (some policies may
    expose none) is simply not drawn, and the nodes still are.
    """
    acts = [np.asarray(a, dtype=float).ravel() for a in activations]
    sizes = [len(a) for a in acts]
    x, y, w, h = rect
    # Leave room for the text labels left of the inputs and right of the outputs.
    label_w = rl.measure_text("heading", LABEL_SIZE) + 16
    output_w = rl.measure_text("throttle -0.00", LABEL_SIZE) + 16
    layout = node_layout(sizes, (x + label_w, y, w - label_w - output_w, h))
    gap = min(h / max(sizes), MAX_NODE_GAP)
    radius = float(np.clip(0.36 * gap, 2.5, 7.0))

    for k, matrix in enumerate(weights[:len(sizes) - 1]):
        matrix = np.asarray(matrix)
        if matrix.shape != (sizes[k], sizes[k + 1]):
            continue
        scale = np.abs(matrix).max() or 1.0
        # Weakest first, so the strongest edges are drawn on top.
        for i, j, wt in reversed(top_edges(matrix, EDGES_PER_LAYER)):
            base = POSITIVE if wt > 0 else NEGATIVE
            alpha = int(EDGE_ALPHA_MAX * (abs(wt) / scale) ** 1.5)
            a, b = layout[k][i], layout[k + 1][j]
            rl.draw_line_ex((float(a[0]), float(a[1])), (float(b[0]), float(b[1])),
                            1.0 + 1.5 * abs(wt) / scale, base + (alpha,))

    for layer, values in zip(layout, acts):
        for (nx, ny), value in zip(layer.tolist(), values):
            rl.draw_circle_v((nx, ny), radius + 1.5, NODE_RIM)
            rl.draw_circle_v((nx, ny), radius, _activation_color(value))

    half = LABEL_SIZE // 2
    if sizes[0] > 3:   # rays + speed, steer, heading: the layout input_labels names
        for (nx, ny), name in zip(layout[0].tolist(), input_labels(sizes[0])):
            tw = rl.measure_text(name, LABEL_SIZE)
            rl.draw_text(name, int(nx - radius - 6 - tw), int(ny - half), LABEL_SIZE, LABEL)
    if sizes[-1] == len(OUTPUT_LABELS):
        for (nx, ny), name, value in zip(layout[-1].tolist(), OUTPUT_LABELS, acts[-1]):
            tx = int(nx + radius + 6)
            rl.draw_text(name, tx, int(ny - half), LABEL_SIZE, LABEL)
            rl.draw_text(f"{value:+.2f}", tx + rl.measure_text("throttle ", LABEL_SIZE),
                         int(ny - half), LABEL_SIZE, _activation_color(value))
