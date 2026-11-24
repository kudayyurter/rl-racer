# Step budgets

Every method gets the same `env_steps_budget` on a given track (spec). It counts car-decisions
(one agent step of one car). Sized so the slowest method (SB3) finishes a main run in about 15 min
on the dev machine (Ryzen 7 7700X, 16 threads), measured 2026-11-23.

## Throughput (env_steps / wall_time_s, ~60 s runs, one at a time)

| method | twisty | oval |
|---|---|---|
| evolution | 6.9k/s | 12.0-12.4k/s * |
| reinforce | 12.6k/s | 12.5k/s |
| a2c | 8.6k/s | 3.9-6.0k/s * |
| ppo | 5.3k/s | 5.3-5.6k/s * |
| sb3 | 5.0k/s (slowest) | 5.0-5.2k/s * |
| ppo_random (eval on twisty + 10 held-out tracks) | 3.0k/s | - |

Before the main runs only REINFORCE was measured on the oval, and it matched twisty, so the oval
budget assumed the others would too. \* The starred oval figures were filled in afterwards from
the main oval runs (lessons 05, 06 and 10; run directories in lesson 10's oval section):
`env_steps / wall_time_s` at the first `metrics.csv` row past 60 s, three seeds each. Those runs
shared the machine (evolution three at a time, the others with about five other runs), so they
are not "one at a time" like the rest of the table. Over whole runs, evaluations included, they
averaged 9.0-10.5k/s (evolution), 8.2-10.3k (REINFORCE), 5.3-5.7k (A2C), 5.3k (PPO) and 4.9-5.2k
(SB3).
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
with SB3 batched it does (4.5M is ~900 s), so the equal-budget oval runs use 4.5M.

Outcome at 4.5M (lessons 05 and 10): REINFORCE's sampled policy laps the oval on every seed, but
its deterministic mean, which every evaluation drives, lapped on only two seeds (first at 1,071,712
and 2,124,672 steps) and crashed in all three final evaluations. The budget was not raised: lesson
05 traces the gap to the mean never being trained as a driver (its `log_std` barely moved in ~140
updates), not to too few steps, and lesson 10 reports the oval bar for REINFORCE as a spec
deviation.

Timing caveat: the ~900 s for sb3 comes from the first ~60 s of a run. Evaluation episodes get longer as
policies improve, so full runs may finish somewhat past 15 min.

Lesson experiments are separate (<= ~5 min each) and set their own budget per experiment.
