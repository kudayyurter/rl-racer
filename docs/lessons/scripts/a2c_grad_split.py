"""Lesson 06: how big are the actor's and the critic's gradients, and how often does the clip bite?

Loads an A2C checkpoint (actor and critic), collects rollouts with it exactly as training does
(``collect_rollout``, truncation bootstrap, ``nstep_returns``, ``a2c_loss``) and, without
stepping, measures per rollout: the gradient norm of the actor's parameters, of the critic's
parameters (already multiplied by ``value_coef``), the combined norm, and the factor
``clip_grad_norm_`` would scale everything by (1 = no clipping). Repeated for each value_coef
given, on the same rollouts.

Run: uv run python docs/lessons/scripts/a2c_grad_split.py runs/a2c-<timestamp>-s0 latest 0.25 0.5 1.0
"""

import json
import sys

import numpy as np
import torch

from racer.algos.a2c import a2c_loss, add_truncation_bootstrap, collect_rollout, nstep_returns
from racer.config import EnvConfig, from_dict
from racer.nets_torch import Critic, GaussianActor
from racer.sim import BatchSim
from racer.track import load_track

N_ROLLOUTS = 20


def norm(params) -> float:
    return float(torch.sqrt(sum((p.grad ** 2).sum() for p in params)))


def main() -> None:
    run, which, coefs = sys.argv[1], sys.argv[2], [float(c) for c in sys.argv[3:]]
    config = json.loads(open(f"{run}/config.json").read())
    env, train = from_dict(EnvConfig, config["env"]), config["train"]
    name = "latest.pt" if which == "latest" else f"ckpt_{int(which):05d}.pt"
    ckpt = torch.load(f"{run}/checkpoints/{name}", weights_only=False)
    actor = GaussianActor(ckpt["obs_dim"], 2, tuple(ckpt["arch"]["hidden"]))
    critic = Critic(ckpt["obs_dim"], tuple(ckpt["arch"]["hidden"]))
    actor.load_state_dict(ckpt["weights"])
    critic.load_state_dict(ckpt["critic_weights"])
    torch.manual_seed(0)
    torch.set_num_threads(2)
    sim = BatchSim(load_track(train["track"]), env, n_cars=train["n_cars"], seed=0, auto_reset=True)
    obs = sim.reset()
    for _ in range(50):  # drive a little first, so cars are spread over the track as in training
        _, obs = collect_rollout(sim, actor, critic, obs, train["n_steps"])

    stats = {c: [] for c in coefs}
    for _ in range(N_ROLLOUTS):
        batch, obs = collect_rollout(sim, actor, critic, obs, train["n_steps"])
        dones = batch["terminated"] | batch["truncated"]
        rewards = add_truncation_bootstrap(batch["rewards"], batch["truncated"],
                                           batch["final_values"], train["gamma"])
        returns = torch.as_tensor(nstep_returns(rewards, dones, batch["last_values"],
                                                train["gamma"]).reshape(-1), dtype=torch.float32)
        b_obs = torch.as_tensor(batch["obs"].reshape(-1, batch["obs"].shape[-1]), dtype=torch.float32)
        b_act = torch.as_tensor(batch["actions"].reshape(-1, 2), dtype=torch.float32)
        for c in coefs:
            actor.zero_grad()
            critic.zero_grad()
            dist = actor.dist(b_obs)
            loss, _ = a2c_loss(dist.log_prob(b_act).sum(-1), critic(b_obs), returns,
                               dist.entropy().sum(-1), c, train["entropy_coef"])
            loss.backward()
            a, v = norm(actor.parameters()), norm(critic.parameters())
            total = float(np.hypot(a, v))
            stats[c].append((a, v, total, min(1.0, train["max_grad_norm"] / total)))

    print(f"{run} {name}: {N_ROLLOUTS} rollouts of {train['n_cars']} x {train['n_steps']}, "
          f"max_grad_norm {train['max_grad_norm']}")
    for c in coefs:
        s = np.array(stats[c])
        print(f"  value_coef {c:4}: |actor grad| {np.median(s[:, 0]):6.3f}, "
              f"|critic grad| {np.median(s[:, 1]):6.3f}, total {np.median(s[:, 2]):6.3f} (medians); "
              f"clipped in {np.mean(s[:, 3] < 1):4.0%} of rollouts, median scale {np.median(s[:, 3]):.3f}")


if __name__ == "__main__":
    main()
