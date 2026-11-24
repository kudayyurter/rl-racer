# 00 · Overview: the project and how to run it

RL Racer is a top-down racing game in which cars learn to drive by trial and error. Every
learning method is written from scratch and kept small enough to read: neuroevolution in numpy,
then REINFORCE, A2C and PPO in PyTorch, with Stable-Baselines3's PPO as an answer key on the same
simulator. These lessons walk through the finished code in the order it builds up: the track,
the car, the environment, then each learning method.

How to use them:

- Read with the code open. Every lesson links to the exact lines it talks about.
- Each lesson ends with experiments: a change to make, a question to predict, a command to run.
  Write your prediction down before opening **What we got**, which holds Claude's measured
  result and says where it came from (a script or a `runs/` directory).
- Runs are seeded but not bit-identical, and timings depend on what else the machine is doing.
  Your numbers will be close to the ones quoted, not equal.

## Setup

Python 3.12 and every dependency are pinned through [uv](https://docs.astral.sh/uv/). PyTorch
comes from the CPU-only index (the networks are tiny; a GPU would not help).

```sh
uv sync          # creates .venv with everything, including the `racer` command
uv run racer --help
```

Always run things through `uv run`, never the system Python.

## Project map

| module | what it does |
|---|---|
| [`racer/config.py`](../../racer/config.py) | `EnvConfig` and `CarParams` dataclasses and strict TOML loading: an unknown key is an error, never silently ignored |
| [`racer/geometry.py`](../../racer/geometry.py) | vectorized ray-to-wall distances and segment crossings (lesson 01) |
| [`racer/track.py`](../../racer/track.py) | track files to a smooth centerline, walls, and the projection that measures progress (lesson 01) |
| [`racer/trackgen.py`](../../racer/trackgen.py) | random valid tracks for the generalization stage (lesson 09) |
| [`racer/car.py`](../../racer/car.py) | batched kinematic bicycle with a grip limit (lesson 02) |
| [`racer/sim.py`](../../racer/sim.py) | `BatchSim`: N cars, sensors, reward, laps, episode ends, resets (lesson 03) |
| [`racer/gym_env.py`](../../racer/gym_env.py) | `RacerEnv`: one car as a standard Gymnasium environment (lesson 03) |
| [`racer/bench.py`](../../racer/bench.py) | simulator throughput benchmark (`racer bench`) |
| [`racer/nets_np.py`](../../racer/nets_np.py) | a whole population of small numpy networks evaluated at once (lesson 04) |
| [`racer/nets_torch.py`](../../racer/nets_torch.py) | the Gaussian policy and critic networks shared by REINFORCE, A2C and PPO |
| [`racer/algos/evolution.py`](../../racer/algos/evolution.py) | neuroevolution: keep the best networks, mutate, repeat (lesson 04) |
| [`racer/algos/reinforce.py`](../../racer/algos/reinforce.py) | REINFORCE, the plainest policy gradient (lesson 05) |
| [`racer/algos/a2c.py`](../../racer/algos/a2c.py) | A2C: REINFORCE plus a critic and bootstrapped returns (lesson 06) |
| [`racer/algos/ppo.py`](../../racer/algos/ppo.py) | PPO: A2C plus GAE and the clipped objective (lesson 07) |
| [`racer/algos/sb3_ppo.py`](../../racer/algos/sb3_ppo.py) | Stable-Baselines3 PPO with the same settings, the answer key (the only file that imports SB3) |
| [`racer/runlog.py`](../../racer/runlog.py) | run directories: config, metrics CSV, checkpoints and ghost laps, written atomically |
| [`racer/policy.py`](../../racer/policy.py) | load any method's checkpoint as a `Policy` (observation in, action out) |
| [`racer/evaluate.py`](../../racer/evaluate.py) | one deterministic evaluation episode: return, laps, best lap and its trace |
| [`racer/compare.py`](../../racer/compare.py) | learning curves and a summary table across runs (`racer compare`) |
| [`racer/drive.py`](../../racer/drive.py) | drive mode: you at the keyboard, on the real simulator |
| [`racer/watch.py`](../../racer/watch.py) | watch mode: a separate window that follows a training run live |
| [`racer/render/camera.py`](../../racer/render/camera.py) | the one mapping from world meters to screen pixels |
| [`racer/render/track_draw.py`](../../racer/render/track_draw.py) | draws the track once into a texture (grass, asphalt, kerbs, start line) |
| [`racer/render/car_draw.py`](../../racer/render/car_draw.py) | cars, shadows and ghost replays |
| [`racer/render/overlay.py`](../../racer/render/overlay.py) | the `D`-key debug overlay (walls, centerline, rays) and HUD panels |
| [`racer/render/netpanel.py`](../../racer/render/netpanel.py) | the live drawing of the policy network in watch mode |
| [`racer/render/curve.py`](../../racer/render/curve.py) | small learning-curve charts from `metrics.csv` |
| [`racer/render/app.py`](../../racer/render/app.py) | the window loop shared by drive and watch, and `--screenshot` |
| [`racer/cli.py`](../../racer/cli.py) | the `racer` command and its subcommands |

Data lives in [`tracks/`](../../tracks) (track JSON) and [`configs/`](../../configs) (one TOML
per method plus [`env.toml`](../../configs/env.toml) for the environment). Training output goes
to `runs/`, which git ignores.

## How the pieces connect

```
  tracks/*.json          configs/env.toml
       |                       |
       v                       v
  track.py  ----------->  sim.py: BatchSim  <----- car.py (physics)
  (centerline, walls,     N cars stepped together; geometry.py (rays, crashes)
   projection)            obs, reward, terminated / truncated
                               |
        +----------------------+-------------------------+
        |                      |                         |
        v                      v                         v
  algos/evolution.py    algos/reinforce.py        gym_env.py / BatchSimVecEnv
  (nets_np.py)          algos/a2c.py, ppo.py      -> algos/sb3_ppo.py
                        (nets_torch.py)
        |                      |                         |
        +----------------------+-------------------------+
                               |  runlog.py (every file written atomically)
                               v
            runs/<method>-<date>-<time>-s<seed>/
                config.json   metrics.csv   checkpoints/latest.pt   ghosts/*.npz
                    |                          |                         |
          +---------+-----------+--------------+-------------------------+
          |                     |
          v                     v
   racer compare          racer watch (separate process, read-only)
   compare.py             watch.py: policy.py loads latest.pt, drives it in a BatchSim,
   -> plots, summary.md   replays ghosts, draws the network and curves (render/*)
```

The important property: training never waits for the viewer. The trainer writes files; the
viewer only reads them, and every file is written to a temporary name and then renamed into
place, so the viewer never sees half a checkpoint.

One unit runs through everything: an **env step** is one decision of one car. Every method gets
the same budget of env steps per track (`env_steps_budget` in its config, see
[`configs/budgets.md`](../../configs/budgets.md)), so learning curves compare fairly no matter how
each method batches its cars.

## Commands

All commands are subcommands of `racer` ([`cli.py`](../../racer/cli.py#L10-L17)); `uv run racer
<command> --help` lists every flag.

| command | what it does | example |
|---|---|---|
| `drive` | drive yourself (Up/W throttle, Down/S brake, Left/A and Right steer, R restart, D overlay, Esc quit) | `uv run racer drive --track oval` |
| `train <method>` | train headless at full speed; method is `evolution`, `reinforce`, `a2c`, `ppo` or `sb3` | `uv run racer train evolution --set track=oval --set env_steps_budget=500000` |
| `watch` | open a viewer on a run directory; it reloads the newest checkpoint every few seconds | `uv run racer watch runs/evolution-<timestamp>-s0` |
| `compare` | learning-curve plots and a summary table for finished runs | `uv run racer compare runs/evolution-* --out results/raw --name mine` |
| `bench` | simulator throughput at several batch sizes | `uv run racer bench` |
| `gen-track` | write a random valid track (same seed, same track) | `uv run racer gen-track --seed 7 --out tracks/rand_7.json` |

Useful flags:

- `train`: `--seed N` (default from the config, 0), `--config PATH` (default
  `configs/<method>.toml`), `--set KEY=VALUE` to override a method setting,
  `--env-config PATH` (default `configs/env.toml`) and `--env-set KEY=VALUE` for environment
  settings, e.g. `--env-set car.max_lat_accel=30`. Both `--set` flags repeat.
- `drive`: `--track NAME_OR_PATH` (default twisty), `--overlay` to start with the overlay on.
  Drive reads `configs/env.toml` directly, so to drive with other car settings, edit that file.
- `watch`: `--track` to drive the policy on another track than it trained on, `--ghosts N`,
  `--reload-s S`.
- `drive` and `watch`: `--screenshot PATH [--frames N]` renders N frames, saves a PNG and exits.
- `bench`: `--seconds S` per batch size.

A file written by `gen-track` drives like any other: `uv run racer drive --track tracks/rand_7.json`.

Overrides go through the same strict loader as the files
([`config.py#L145-L166`](../../racer/config.py#L145-L166)), so a typo stops the run before it
starts instead of quietly using a default (Experiment 3).

## Tests

```sh
uv run pytest -q                                                    # the fast suite
uv run pytest tests/test_track.py -v                                # one file
uv run pytest tests/test_track.py::test_projection_search_is_windowed -v   # one test
```

On 2026-11-23 the fast suite was 190 tests, all passing, in 18.8 s. Tests marked `slow` (real
training, minutes long) are skipped by default and run with `uv run pytest -m slow`; at the time
of writing there are none. The tests are worth reading as documentation: each lesson ends with a
table of the ones that pin its key behaviours.

## Reading order

| lesson | topic |
|---|---|
| **00** | this overview: map, data flow, commands, tests |
| [**01**](01-track-and-geometry.md) | track and geometry: the spline, arc-length samples, windowed projection, ray and crossing math |
| [**02**](02-car-physics.md) | car physics: the kinematic bicycle, the grip limit and why it forces braking |
| [**03**](03-sensors-and-env.md) | sensors and the environment: observation, reward, terminated vs truncated, batching, Gymnasium |
| [**04**](04-neural-nets-and-evolution.md) | neural nets and neuroevolution: a forward pass by hand, learning with no gradients |
| [**05**](05-reinforce.md) | REINFORCE: the policy-gradient theorem derived, and why it is noisy |
| [**06**](06-a2c.md) | A2C: a learned baseline, n-step bootstrapping, the truncation fix |
| [**07**](07-ppo.md) | PPO: the clipped objective, GAE, minibatch epochs, checked against Stable-Baselines3 |
| [**08**](08-fastest-lap.md) | fastest lap: reward shaping, what a time cost pays for under each episode rule, and why a bigger one made neither method faster (evolution stopped lapping) |
| **09** | generalization: training on random tracks, testing on unseen ones (coming) |
| **10** | results: every method compared by env steps and by wall-clock time (coming) |

## What to look for in the viewer

`racer watch` on a run shows (see [`docs/img/watch.png`](../img/watch.png)): on the left, the
current policy driving live plus faded ghost cars replaying earlier best laps; on the right, the
network firing as it drives (inputs on the left, outputs `steer` and `throttle` on the right),
the best evaluation lap and evaluation progress over training, and a status block saying which
checkpoint is driving. A new checkpoint takes over only when the live car's episode ends, so a lap
is never driven half by one policy and half by another.

## Experiments

### 1. Your first training run

**Change.** Nothing: train neuroevolution on the easy oval with a small budget of 500,000 env
steps, then watch the result.

**Predict.** Evolution evaluates its best network every 5 generations. Will the first evaluation
already complete a lap? How long will the run take?

**Run.**

```sh
uv run racer train evolution --set track=oval --set env_steps_budget=500000
uv run racer watch runs/evolution-<the timestamp it printed>-s0
```

<details><summary>What we got</summary>

Run `runs/evolution-20261123-193221-s0` (seed 0), 2026-11-23. Console output:

```
evolution on oval: population 128, elite 16, network (12, 32, 2) (482 weights)
gen    1  steps   111,354  best    37.82  mean    -9.65
gen    2  steps   179,140  best    53.95  mean    -3.77
gen    3  steps   237,343  best    53.95  mean     0.82
gen    4  steps   319,863  best   150.80  mean     2.26
gen    5  steps   407,810  best   153.05  mean     9.67  | eval laps 2  progress 1428.6 m  best lap 28.72 s
gen    6  steps   507,286  best   153.42  mean    18.11  | eval laps 2  progress 1427.9 m  best lap 28.45 s
```

The first evaluation (generation 5, 407,810 steps, 38.2 s of wall time) already drove both laps.
The jump happens at generation 4, where the best car's return goes from 54 to 151: one network
found how to get round the turns. The `mean` column is the whole population and lags far behind,
because most children are mutated copies that crash. The run took 46.4 s in total
(`wall_time_s` in its `metrics.csv`; the budget is checked between generations, so it overshoots a
little to 507,286 steps). In the viewer, the car laps with a best evaluation lap of 28.450 s.

</details>

### 2. Same run, another seed

**Change.** Add `--seed 1`, then compare both runs.

**Predict.** Will seed 1 lap at the same generation? Will its best lap match seed 0's to within
a second?

**Run.**

```sh
uv run racer train evolution --set track=oval --set env_steps_budget=500000 --seed 1
uv run racer compare runs/evolution-<seed 0 run> runs/evolution-<seed 1 run> --out results/raw --name seeds
```

<details><summary>What we got</summary>

Runs `runs/evolution-20261123-193221-s0` and `runs/evolution-20261123-193617-s1`, 2026-11-23.
Summary table written by `racer compare`:

```
| group | seeds | success | best lap s, median (range) | env steps to first lap | wall s to first lap |
|---|---|---|---|---|---|
| evolution · oval | 2 | 2/2 | 28.97 (28.45–29.48) | 412,176 | 37.9 |
```

Both seeds lapped at their first evaluation (generation 5: 407,810 steps for seed 0, 416,542 for
seed 1). The best laps differ by about a second: 28.45 s and 29.48 s. The two seeds reached a
working driver along different paths: seed 1's best return jumped at generation 2, seed 0's at
generation 4. That spread is why the main results use three seeds and report ranges.

</details>

### 3. A typo in an override

**Change.** Misspell a setting on purpose: `mutation_sdt` instead of `mutation_std`.

**Predict.** Does the run start with the default mutation size, or stop?

**Run.**

```sh
uv run racer train evolution --set mutation_sdt=0.2
```

<details><summary>What we got</summary>

Run 2026-11-23. It stops before creating a run directory; the last line of the traceback is:

```
racer.config.ConfigError: unknown config key(s) for EvolutionConfig: mutation_sdt
```

A silently ignored typo would cost a whole training run and leave a wrong conclusion ("mutation
size made no difference"). The check is in [`from_dict`](../../racer/config.py#L85-L113).

</details>

Next: [01 · Track and geometry](01-track-and-geometry.md).
