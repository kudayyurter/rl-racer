"""The Policy interface and checkpoint loading.

Why this exists: the viewer and ``evaluate`` must drive a car with *any* method's trained
policy (numpy evolution nets, torch actor networks, an SB3 model) without knowing which. They
only need the small ``Policy`` protocol below. Each method's module registers a loader that
turns its checkpoint dict into a Policy; ``load_policy`` reads the file and picks the loader
from the checkpoint's ``method`` field.

Method modules are imported lazily, only when a checkpoint of that method is loaded, so
viewing an evolution run does not import torch networks or Stable-Baselines3.
"""

import importlib
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
import torch


class Policy(Protocol):
    """A trained controller. Observations are (N, D), actions are (N, 2) = [steer, throttle]."""

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic actions in [-1, 1] (the mean, no exploration noise)."""

    def activations(self, obs: np.ndarray) -> list[np.ndarray]:
        """For one observation (1, D): the input, each hidden layer, then the output."""

    def layer_weights(self) -> list[np.ndarray]:
        """Weight matrices, for drawing the network in the viewer."""


Loader = Callable[[dict], Policy]

_LOADERS: dict[str, Loader] = {}

# Which module defines the loader for each method. Importing it runs its register_loader call.
_METHOD_MODULES = {
    "evolution": "racer.algos.evolution",
    "reinforce": "racer.algos.reinforce",
    "a2c": "racer.algos.a2c",
    "ppo": "racer.algos.ppo",
    "sb3": "racer.algos.sb3_ppo",
}


def register_loader(method: str, fn: Loader) -> None:
    """Declare how to build a Policy from a checkpoint dict whose ``method`` is ``method``."""
    _LOADERS[method] = fn


def load_policy(path: str | Path) -> Policy:
    """Read a checkpoint file and build the Policy its method describes."""
    # weights_only=False: our checkpoints hold plain dicts with numpy arrays and config dicts,
    # which the safe loader rejects. We only ever load files this project wrote itself.
    ckpt = torch.load(path, weights_only=False)
    method = ckpt["method"]
    if method not in _LOADERS and method in _METHOD_MODULES:
        importlib.import_module(_METHOD_MODULES[method])
    if method not in _LOADERS:
        known = ", ".join(sorted(_METHOD_MODULES))
        raise ValueError(f"unknown checkpoint method {method!r} (known methods: {known})")
    return _LOADERS[method](ckpt)
