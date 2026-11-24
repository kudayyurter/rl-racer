# 03 · Sensors and the environment

Lessons 01 and 02 built a track and a car. This lesson turns them into a reinforcement-learning
*environment*: something a learner can repeatedly hand an action to and get back what the car
now sees, a reward, and whether the episode is over. One class does all of it,
[`BatchSim`](../../racer/sim.py#L65-L261), and every method in the project trains against it.
So the rules written here (what the car observes, what earns reward, when an episode ends) are
literally what every learner optimizes.

Files: [`racer/sim.py`](../../racer/sim.py), [`racer/gym_env.py`](../../racer/gym_env.py),
[`racer/bench.py`](../../racer/bench.py), [`tests/test_sim.py`](../../tests/test_sim.py),
[`tests/test_gym_env.py`](../../tests/test_gym_env.py).

## The words, briefly

- **Observation**: the numbers the policy gets each decision (here 12 floats).
- **Action**: what the policy sends back (`[steer, throttle]`).
- **Reward**: one number per decision saying how good that step was.
- **Episode**: one run from spawn until a crash, a time limit or enough laps.
- **Return**: the (discounted) sum of rewards from some step to the end of the episode,
  `G_t = r_t + γ r_{t+1} + γ² r_{t+2} + ...`. Every learner here is, one way or another, trying
  to pick actions that make the return large.
- **Policy**: the function from observation to action. That is what gets trained.

## One decision = several physics steps

A policy does not act every physics step. [`step`](../../racer/sim.py#L180-L243) runs
`action_repeat` (default 4) physics substeps with the same action, so the policy decides 15 times
per simulated second. Per substep ([lines 192-218](../../racer/sim.py#L192-L218)):

1. move every car ([`step_cars`](../../racer/car.py#L45-L75));
2. project each car onto the centerline and add the change in `s` to its progress (lesson 01);
3. count a lap if progress passed another full track length;
4. check the car's edges against the nearby walls for a crash.

Then, once per decision, cast the rays and build the observation.

Why repeat the action? Fifteen decisions per second is plenty to drive a car. It cuts the number of network evaluations by 4. And it shortens the problem *in decisions*:
a 90 s episode is 1,350 decisions instead of 5,400. That matters for learning, because the
discount `γ = 0.99` used by the gradient methods (in [`configs/ppo.toml`](../../configs/ppo.toml) and the
others) counts per decision: a reward 100 decisions away is weighted `0.99^100 ≈ 0.37`. With
`action_repeat = 4` that is 6.7 s of driving ahead, enough to see a corner coming. With
`action_repeat = 1` the same `γ` would look only 1.7 s ahead. Crash and lap checks still run every
substep, so nothing is missed between decisions; only the policy's reaction time is coarser.
Experiment 2 measures what it does to simulator speed.

## The observation

[`_observe`](../../racer/sim.py#L247-L261) builds `n_rays + 3` numbers per car
([`obs_dim`](../../racer/sim.py#L42-L44)):

| entries | value | scaled by | range |
|---|---|---|---|
| 0 .. 8 | distance along each ray to the first wall | `/ ray_range` (60 m) | [0, 1] |
| 9 | speed | `/ max_speed` (50 m/s) | [0, 1] |
| 10 | current front-wheel angle | `/ max_steer` (0.6 rad) | [-1, 1] |
| 11 | heading minus the track direction, wrapped to [-π, π) | `/ π` | [-1, 1) |

The rays fan evenly over `ray_spread_deg = 180` degrees around the heading
([sim.py#L85-L87](../../racer/sim.py#L85-L87)); ray 0 points to the car's right, the last one to
its left. [`_cast_rays`](../../racer/sim.py#L264-L272) calls the vectorized `ray_distances` from
lesson 01 with a wall window wide enough to cover the full 60 m.

Why these inputs and not others:

- **Rays** are the car's eyes. They describe the road around the car *relative to the car*, which
  is what a driving decision depends on. They say nothing about where on the track the car is,
  deliberately: a policy that sees only its surroundings can, in principle, drive a track it has
  never seen (lesson 09 tests that).
- **Speed.** One frame of rays looks the same at 5 m/s and at 40 m/s, but the right action is
  completely different (lesson 02: brake or not). Without speed the policy cannot solve the
  braking problem at all.
- **Steer angle.** The wheel cannot jump (steer-rate limit), so where it is now shapes what the
  car does next. Leaving it out would hide part of the car's state from the policy.
- **Heading error** says whether the car points along the road. On a wide straight the rays can
  be nearly symmetric whether the car is aligned or slightly crooked; this resolves it directly.

The general principle: the observation should contain what is needed to predict what happens
next (the *Markov* property). Position on the track and lap time are left out on purpose, as
above.

Why scale everything to about [-1, 1]? The networks' weights are initialized for inputs of about
unit size (Xavier scaling in [`nets_np.py`](../../racer/nets_np.py#L26-L27), PyTorch's default for
`nn.Linear`), and their tanh units saturate (stop responding) when the weighted sum gets large. A raw ray of 60 alongside a raw
steer angle of 0.3 would let the rays swamp everything else at the start of training. Scaled
inputs keep every input in the range where the network is sensitive, and match the
`Box(-1, 1)` observation space the Gymnasium wrapper declares.

For a concrete example, [`docs/lessons/scripts/start_observation.py`](scripts/start_observation.py)
prints the observation a car gets on twisty's start line:

```
entry  meaning                             raw     obs
    0  ray at -90.0 deg                 6.00 m   0.100
    1  ray at -67.5 deg                 6.49 m   0.108
    2  ray at -45.0 deg                 8.49 m   0.141
    3  ray at -22.5 deg                15.68 m   0.261
    4  ray at +0.0 deg                 60.00 m   1.000
    5  ray at +22.5 deg                15.68 m   0.261
    6  ray at +45.0 deg                 8.49 m   0.141
    7  ray at +67.5 deg                 6.49 m   0.108
    8  ray at +90.0 deg                 6.00 m   0.100
    9  speed / max_speed              0.00 m/s   0.000
   10  steer / max_steer              0.00 rad   0.000
   11  heading error / pi             0.00 rad   0.000
```

The sideways rays read 6 m (half of the 12 m road), the diagonal ones `6 / sin(angle)`, and the
forward ray sees nothing within 60 m. A symmetric picture: the car is centered and aligned.

## The reward

[Lines 225-226](../../racer/sim.py#L225-L226):

```
reward = progress_scale * progress - time_cost + lap_bonus * lapped - crash_penalty * crashed
```

| term | default ([`configs/env.toml`](../../configs/env.toml)) | what it is for |
|---|---|---|
| `progress_scale * progress` | 0.1 per meter driven along the track this decision | the main signal: a dense reward every step for moving forward. Driving backwards makes it negative |
| `- time_cost` | 0.01 per decision | standing still is now slightly negative, so parking is never the safest strategy |
| `+ lap_bonus` | 10 per completed lap | a clear marker for "finished a lap" |
| `- crash_penalty` | 10 on the decision the car crashes | crashing costs the equivalent of 100 m of progress, on top of losing all the progress it would have made |

Some scale: one lap of the oval is 713.64 m, worth 71.4 in progress reward. At 20 m/s the car
earns about 2.0 progress reward per second, against a time cost of 0.15 per second
(15 decisions × 0.01). The time cost is small next to progress; its job is to remove the
"do nothing" option, not to drive the lap time. Lesson 08 raises it and shows what that changes.

Why progress and not just "+1 per lap"? A reward that only arrives after a full lap gives a
random policy nothing to learn from: it crashes long before it ever sees a lap. Progress rewards
every meter, so the first policy that drives 20 m instead of 10 m is already measurably better.

Why is progress measured from where the car spawned and not from `s = 0`? On the oval the start
line is at `s = 50 m` (`start_index = 1` in the track file), and with `random_start` cars spawn
anywhere. So [`_spawn_at`](../../racer/sim.py#L155-L176) projects the new car once and zeroes
`traveled`; each substep adds the wrapped change in `s`
([lines 199-207](../../racer/sim.py#L199-L207)): crossing `s = 0` forwards is a small positive
step, not `-length`.

A **lap** is counted when `traveled` passes another whole track length
([lines 209-214](../../racer/sim.py#L209-L214)). Driving back and forth across the start line
therefore never counts: going back subtracts exactly what going forward added.
[`test_start_line_oscillation_counts_no_lap`](../../tests/test_sim.py#L104-L117) teleports a car
3 m either side of the line 60 times, both at the real start line and at `s = 0` where `s` wraps,
and asserts no lap.

## How an episode ends: terminated vs truncated

[Lines 220-223:](../../racer/sim.py#L220-L223)

- **terminated**: a real ending, the car failed. Either it crashed, or it drove more than
  `wrong_way_m` (15 m) backwards from the furthest point it reached.
- **truncated**: the episode was cut short but the car did nothing wrong. Either the time limit
  (`max_episode_s`, 90 s) ran out, or it finished `max_laps` (2) laps.

Terminated wins if both happen on the same decision, and each ending is reported exactly once.
The distinction matters for every method that learns a value function (A2C, PPO, SB3). Such a
method trains a critic `V(s)` to predict the return from state `s`, using the bootstrapped target

```
target(s_t) = r_t + γ · V(s_{t+1})
```

- If the car **crashed**, nothing follows. `V(s_{t+1})` should be 0, so the target is just `r_t`.
- If the episode was **truncated**, the car could have kept driving and scoring. The future is
  still there; we only stopped watching. The correct target keeps `γ · V(s_{t+1})`, where
  `s_{t+1}` is the state the car was really in.

Treating a time limit as a crash teaches the critic that a perfectly good state, mid-straight at
full speed, is worth nothing whenever it happens to fall at the 90 s mark. The critic cannot even
tell when that is: time is not in the observation, so the same-looking state gets a value of 0 on
one visit and 40 on another. That is noise in the critic's targets, and through the critic, in
the policy update. Ending a run after two laps is worse: the finishing lap would look like a
crash.

That is why `StepResult` carries `final_obs` as well as `obs`
([lines 52-62](../../racer/sim.py#L52-L62)). With auto-reset, `obs` for a finished car is already
its *respawn* observation; `final_obs` is where it actually ended, the `s_{t+1}` the truncation
bootstrap needs. A2C and PPO add `γ · V(final_obs)` to the reward of truncated steps
([a2c.py#L113-L116](../../racer/algos/a2c.py#L113-L116),
[ppo.py#L145-L148](../../racer/algos/ppo.py#L145-L148)); SB3 does the same from the
`TimeLimit.truncated` flag. REINFORCE deliberately does not, and lesson 05 discusses the bias
that leaves.

Why does wrong way *terminate*? A car that turned around would otherwise drive backwards until the
time limit, burning simulation on useless experience. Its negative progress is already its
penalty, so the rule adds no extra one: it just stops the episode. 15 m (rather than 0) gives a
car room to wobble backwards a little without being cut off.
[`test_wrong_way_terminates`](../../tests/test_sim.py#L141-L154) and
[`test_timeout_truncates`](../../tests/test_sim.py#L157-L165) pin both kinds.

## Batching and auto-reset

All per-car state is numpy arrays of length N ([lines 91-102](../../racer/sim.py#L91-L102)), and
the step path has no Python loop over cars. That is the whole point of `BatchSim`: numpy's
per-call overhead is paid once for all N cars, so stepping 256 cars costs far less than 256
times one car. Experiment 1 measures it.

When a car's episode ends, two behaviours are available ([class docstring](../../racer/sim.py#L65-L72)):

- `auto_reset=False`: the car freezes (`alive` becomes False, reward 0) until `reset()`. That is
  what evolution wants: every car in the population drives exactly one episode, and the
  generation ends when all are done.
- `auto_reset=True`: the car respawns immediately ([lines 233-237](../../racer/sim.py#L233-L237)),
  so every car is always driving. That is what the policy-gradient methods want: they collect
  fixed-length chunks of experience from N cars and do not care where episodes begin or end
  (as long as `terminated`, `truncated` and `final_obs` say so).

## The Gymnasium interface

[Gymnasium](https://gymnasium.farama.org/) is the standard Python interface for RL environments:
`reset()` returns `(obs, info)`, `step(action)` returns `(obs, reward, terminated, truncated,
info)`. [`RacerEnv`](../../racer/gym_env.py#L23-L61) is that interface over a `BatchSim` with one
car. It reimplements nothing; it only adapts shapes and types (float32 observations, Python
floats and bools). It exists so that any off-the-shelf RL library can train on exactly our
physics and reward, which is how Stable-Baselines3 serves as the answer key in lesson 07.
[`test_passes_env_checker`](../../tests/test_gym_env.py#L14-L16) runs Gymnasium's own checker,
and [`test_matches_batchsim`](../../tests/test_gym_env.py#L25-L42) checks step-for-step equality
with the raw simulator.

One wrapper detail shows how strict that checker is: `info["lap_time"]` appears only on the step a
lap finishes ([gym_env.py#L56-L59](../../racer/gym_env.py#L56-L59)). A NaN placeholder on every
other step would fail the checker's determinism test, because `NaN != NaN`.

A single-car interface is also slow, for the reason Experiment 1 shows. SB3 trains on 16 cars at
once through a "vector env". The first version gave it 16 separate one-car `RacerEnv`s, each
stepped from Python, and SB3 ran at 1.4k env steps/s. [`BatchSimVecEnv`](../../racer/algos/sb3_ppo.py#L136-L213)
replaced that with one `BatchSim` of 16 cars behind SB3's `VecEnv` interface, the same batched
simulator our own PPO uses, and SB3 went to 5.0k steps/s (both figures from
[`configs/budgets.md`](../../configs/budgets.md), measured 2026-11-23). The algorithm did not
change; the 3.6× was harness overhead (Experiment 1 shows where it comes from). Without that fix, SB3 would have needed about 3.6× the time
for the same step budget, and the speed comparison in lesson 10 would have measured plumbing.

## What to look for in the viewer

`uv run racer drive --overlay`:

- The progress panel counts meters from the start line. Drive one lap and watch `laps` tick over
  exactly when progress wraps.
- Turn around and drive backwards: after 15 m the episode ends ("Crashed (or wrong way): back to
  the start") with no crash, which is the wrong-way rule.
- Rock back and forth over the start line: the lap counter does not move.

Drive mode runs the real `BatchSim`, with two changes that do not touch physics: `action_repeat`
is 1 so the keyboard is read every physics step, and episodes never end by time or laps
([drive.py#L110-L111](../../racer/drive.py#L110-L111)).

## Experiments

### 1. Throughput across batch size: `racer bench`

**Change.** Nothing. [`racer bench`](../../racer/bench.py#L21-L44) steps 1, 16, 64, 128 and 256
cars with random actions on the oval (auto-reset on, so crashes and respawns are included) for
3 s each and reports car-steps per second (one car-step = one decision of one car).

**Predict.** If there were no per-call overhead, all batch sizes would give the same car-steps/s.
How much faster per car do you expect 256 cars to be than 1? Where does the gain level off?

**Run.**

```sh
uv run racer bench
```

<details><summary>What we got</summary>

Two runs of `uv run racer bench`, 2026-11-23, one after the other (desktop apps running in the
background):

```
  cars  decisions  seconds  car-steps/s        cars  decisions  seconds  car-steps/s
     1       4903     3.00        1,634           1       4941     3.00        1,647
    16       2430     3.00       12,960          16       2422     3.00       12,914
    64        947     3.00       20,197          64        943     3.00       20,114
   128        501     3.00       21,373         128        492     3.00       20,961
   256        329     3.00       28,070         256        281     3.00       23,975
```

- One car manages about 1.6k car-steps/s; 16 cars about 12.9k, 8× more for the same Python
  overhead. This explains SB3's old 1.4k steps/s: 16 one-car sims stepped one after another
  together run at about the one-car rate (each still pays the full per-call overhead), so the
  simulator alone capped SB3 near 1.6k. With one batched 16-car sim the simulator could deliver
  about 12.9k, but SB3 reached 5.0k rather than 12.9k because its wall time also includes running
  the networks and the gradient updates.
- Past 64 cars the gain flattens (20k at 64, 21-28k at 256): the numpy arithmetic itself now
  dominates, and it grows with N.
- The 256-car figure moved from 28.1k to 24.0k between two back-to-back runs, a 15% difference.
  Single bench numbers are rough on a machine that is doing other things.

</details>

### 2. action_repeat 2 vs 4 vs 8

**Change.** `racer bench` always uses the default config, so the script
[`docs/lessons/scripts/bench_action_repeat.py`](scripts/bench_action_repeat.py) runs the same
loop with `action_repeat` set to 2, 4 and 8, for 16 and 256 cars. Besides car-steps/s it prints
simulated seconds per wall second (`car-steps/s × action_repeat × dt`), because a decision lasts
longer when the action is repeated more.

**Predict.** Doubling `action_repeat` doubles the physics per decision but not the ray casting
(still once per decision). Will car-steps/s halve, drop less than half, or stay the same? Which
setting covers the most simulated driving per second?

**Run.**

```sh
uv run python docs/lessons/scripts/bench_action_repeat.py      # about 20 s
```

<details><summary>What we got</summary>

Output of `docs/lessons/scripts/bench_action_repeat.py`, run 2026-11-23:

```
 cars action_repeat  car-steps/s  sim-s per wall-s
   16             2       17,829               594
   16             4       12,873               858
   16             8        8,570             1,143
  256             2       28,645               955
  256             4       23,247             1,550
  256             8       17,206             2,294
```

Doubling the repeat costs far less than half the decision rate: with 256 cars, 2 → 4 drops
car-steps/s by 19% and 4 → 8 by 26%, because ray casting and observation building happen once per
decision however many substeps it holds. In simulated driving per wall second, 8 is 2.4× as fast as
2. But step budgets count *decisions* (lesson 00), so a larger `action_repeat` buys more driving
per budget only by giving the policy fewer, coarser decisions: at 8 it reacts every 0.13 s. The
default of 4 is the middle ground. (The 256-car, repeat-4 figure, 23.2k, is inside the range of
the two `racer bench` runs above.)

</details>

### 3. Predict the return of a known lap

**Change.** Nothing. [`docs/lessons/scripts/reward_breakdown.py`](scripts/reward_breakdown.py)
drives one full episode on the oval with the coasting P-controller from lesson 02 and adds up each
reward term separately.

**Predict.** The controller laps the oval (713.64 m) twice at about 20 m/s, and the episode ends
by `max_laps`. Use the reward table to estimate the return: progress for two laps, the lap
bonuses, and a time cost for roughly 75 s at 15 decisions per second. Is the episode terminated or
truncated?

**Run.**

```sh
uv run python docs/lessons/scripts/reward_breakdown.py
```

<details><summary>What we got</summary>

Output of `docs/lessons/scripts/reward_breakdown.py`, run 2026-11-23:

```
oval length 713.64 m; episode truncated after 74.93 s (1124 decisions of 4 physics steps)
  progress  0.1 * 1428.5 m                   +142.85
  time cost -0.01 * 1124 decisions            -11.24
  lap bonus 10.0 * 2 laps                     +20.00
  crash     -10.0 * 0                          -0.00
  sum of terms                               +151.61
  return paid by BatchSim                    +151.61
```

The terms add up exactly to what the simulator paid. Progress is 1428.5 m, slightly more than two
laps (1427.3 m), because the second lap ends partway through a decision and the whole decision's
progress counts. The episode is truncated (two laps), not terminated. For comparison, the
evolution run from lesson 00 (`runs/evolution-20261123-193221-s0`) reached an evaluation return of
153.42 on the same track: the same progress and bonuses, less time cost, because it laps faster.

</details>

## Tests worth reading

| test | what it pins |
|---|---|
| [`test_start_line_oscillation_counts_no_lap`](../../tests/test_sim.py#L104-L117) | wiggling over the start line (or over `s = 0`) never counts a lap |
| [`test_lap_counted_once`](../../tests/test_sim.py#L69-L101) | a real lap counts once, ends at the spawn point, and its time is right |
| [`test_max_laps_truncates_with_lap_bonus`](../../tests/test_sim.py#L120-L138) | finishing the last lap pays the bonus and truncates |
| [`test_auto_reset_returns_final_obs`](../../tests/test_sim.py#L168-L180) | after auto-reset, `final_obs` keeps the pre-reset observation |
| [`test_passes_env_checker`](../../tests/test_gym_env.py#L14-L16) | `RacerEnv` passes Gymnasium's checker |
| [`test_vec_env_matches_racer_env_step_by_step`](../../tests/test_sb3.py#L77-L104) | SB3's batched env and `RacerEnv` agree step by step |

Next: [04 · Neural nets and neuroevolution](04-neural-nets-and-evolution.md).
