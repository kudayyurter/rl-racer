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
| sb3 | 1.4k/s (slowest) | 1.4k/s |
| ppo_random (eval on twisty + 10 held-out tracks) | 3.0k/s | - |

Oval throughput matches twisty for the two methods measured, so oval is assumed equal for the rest.

## Budgets

| run | `env_steps_budget` | slowest method's time |
|---|---|---|
| twisty, all methods (in `configs/{evolution,reinforce,a2c,ppo,sb3}.toml`) | 1_200_000 | sb3 ~860 s |
| oval, all methods (pass `--set env_steps_budget=1200000 --set track=oval`) | 1_200_000 | sb3 ~880 s |
| ppo_random (in `configs/ppo_random.toml`) | 2_500_000 | ~830 s |

Oval note: lessons need REINFORCE to lap the oval on every seed, which may want 2M or more steps
(a 1.5M pilot lapped only at ~1.2M). At 2M steps SB3 would take about 24 min, over the 15 min target,
so the 15 min value (1.2M) is used for the equal-budget oval runs. If REINFORCE does not lap reliably at
1.2M, raise the oval budget for all methods together; do not give one method a different budget.

Lesson experiments are separate (<= ~5 min each) and set their own budget per experiment.
