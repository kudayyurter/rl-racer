# Step budgets

Every method gets the same `env_steps_budget` on a given track (spec). It counts car-decisions
(one agent step of one car). Sized so the slowest method (SB3) finishes a main run in about 15 min
on the dev machine (Ryzen 7 7700X, 16 threads), measured 2026-11-23.

## Throughput (env_steps / wall_time_s, ~60 s runs, one at a time)

| method | twisty | oval |
|---|---|---|
| evolution | 6.9k/s | not measured |
| reinforce | 12.6k/s | 12.5k/s |
| a2c | 8.6k/s | not measured |
| ppo | 5.3k/s | not measured |
| sb3 | 5.0k/s (slowest) | not re-measured |
| ppo_random (eval on twisty + 10 held-out tracks) | 3.0k/s | - |

Oval throughput matched twisty for the methods measured, so oval is assumed equal for the rest.
sb3 was 1.4k/s until it was moved onto one batched `BatchSim` (`BatchSimVecEnv`, same as our PPO's
sim); the 1.4k/s was harness overhead, not the algorithm. The sb3 figure above is measured after that
change (286,720 steps in 57 s); the other rows are unchanged from the first measurement.

## Budgets

| run | `env_steps_budget` | slowest method's time |
|---|---|---|
| twisty, all methods (in `configs/{evolution,reinforce,a2c,ppo,sb3}.toml`) | 4_500_000 | sb3 ~900 s (ppo ~850 s) |
| oval, all methods (pass `--set env_steps_budget=4500000 --set track=oval`) | 4_500_000 | sb3 ~900 s |
| ppo_random (in `configs/ppo_random.toml`) | 2_500_000 | ~830 s |

Oval note: lessons need REINFORCE to lap the oval on every seed, which needed more than 1.2M steps
(a 1.5M pilot lapped only at ~1.2M). The oval budget should be at least 2M if the 15 min rule allows;
with SB3 batched it does (4.5M is ~900 s), so the equal-budget oval runs use 4.5M. If REINFORCE still
does not lap reliably, raise the budget for all methods together; do not give one method its own.

Lesson experiments are separate (<= ~5 min each) and set their own budget per experiment.
