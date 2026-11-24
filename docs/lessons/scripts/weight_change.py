"""Lesson 05: how far did training move the actor's weights from where they started?

Rebuilds a run's *initial* actor exactly as ``train_reinforce`` / ``train_a2c`` do (seed torch
with the run's seed, then construct the ``GaussianActor`` first, before anything else draws from
torch's generator) and compares it with the final checkpoint: the largest and the median absolute
change over all parameters, and the change of ``log_std``. Adam moves each parameter by roughly
``lr`` per step (more when its recent gradients are consistent), so the largest change divided by
``lr * updates`` says how close to that rough ceiling the run came.

Run: uv run python docs/lessons/scripts/weight_change.py runs/reinforce-<timestamp>-s0 ...
"""

import json
import sys
from pathlib import Path

import torch

from racer.nets_torch import GaussianActor


def main() -> None:
    for arg in sys.argv[1:]:
        path = Path(arg)
        config = json.loads((path / "config.json").read_text())
        ckpt = torch.load(path / "checkpoints" / "latest.pt", weights_only=False)
        torch.manual_seed(config["seed"])
        initial = GaussianActor(ckpt["obs_dim"], 2, tuple(ckpt["arch"]["hidden"]))
        start = initial.state_dict()
        deltas = torch.cat([(ckpt["weights"][k] - start[k]).abs().flatten() for k in start])
        lr, updates = config["train"]["lr"], ckpt["iteration"]
        d_log_std = (ckpt["weights"]["log_std"] - start["log_std"]).tolist()
        print(f"{path.name}: {updates} updates at lr {lr:g}; |change| over {deltas.numel():,} "
              f"parameters: max {deltas.max():.4f} ({deltas.max() / (lr * updates):.2f} x lr x "
              f"updates), median {deltas.median():.4f}; log_std change "
              f"{d_log_std[0]:+.4f} (steer) {d_log_std[1]:+.4f} (throttle)")


if __name__ == "__main__":
    main()
