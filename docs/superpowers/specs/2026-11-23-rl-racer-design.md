# RL Racer: design

Date: 2026-11-23
Status: approved by owner 2026-11-23

## Intent

The owner wants to learn how reinforcement learning works, starting from zero.
It is their first neural network. They will not write code. Claude builds the
whole project, and the owner studies it afterwards: reading the code, reading
the lessons, running experiments, and watching cars learn.

The payoff is visual. A top-down car starts by crashing into walls and, lap by
lap, learns to drive clean and fast. Every step from "random network" to
"fast lap" must be traceable to readable code.

## Decisions by the owner

- Top-down 2D racing game drawn with raylib, through the `pyray` Python bindings.
- Everything in Python. No C/C++.
- Learning algorithms are written from scratch, not taken from a library.
- Learning path: neuroevolution first, then policy gradients (REINFORCE, then PPO).
- Track walls are line segments built from a centerline (not a tile grid, not a
  painted image).
- "Clean circuit" look: mowed-stripe grass, asphalt, white edge lines, red/white
  kerbs, checkered start line, car with a drop shadow.
- The "lean" approach (B below).
- Claude builds everything; the owner analyzes after.

## Goals

1. A drivable game: keyboard control, lap timer, best-lap display.
2. A fast headless simulation that steps many cars at once.
3. Three families of learners, each from scratch and readable:
   neuroevolution (numpy), REINFORCE, A2C and PPO (PyTorch).
4. A viewer that shows learning live without slowing training.
5. A lesson document per stage, written for someone reading finished code.
6. Experiments: settings the owner can change and re-run, with a prediction
   to check in each lesson.
7. A reference check against Stable-Baselines3 on the same environment.
8. A final comparison: lap times and learning curves for every method.

## Non-goals (v1)

- In-game track editor. Tracks are hand-written JSON or generated.
- Drift / slip physics. Grippy handling only.
- The "Detailed circuit" look (gravel, trees, grandstand, tyre marks).
- GPU training. The networks are tiny; CPU is faster for them.
- Human-vs-AI racing, multiplayer, sound.

## Anti-goals (failure even if it works)

- Code that is clever but hard to read. Clarity beats speed and brevity.
- A library doing the learning, so there is nothing to read.
- Lessons with invented numbers. Every number in a lesson comes from a real run.
- Results that cannot be reproduced. Every run is seeded.
- Training that only works while the window is open.

## Constraints

- Machine: CachyOS, Ryzen 7 7700X, 30 GB RAM, RTX 5060 Ti 16 GB (unused).
- System Python is 3.14. Use `uv` with a pinned Python (3.12 or whatever
  version `raylib` (raylib-python-cffi), `torch` and `stable-baselines3` all ship
  wheels for) so installs do not break.
- Project lives at `~/DEV/rl-racer`, git repo, owner GitHub `kudayyurter`
  (pushing is not part of this design).

## Approaches considered

- **A. Original plan as first brainstormed.** Game loop that trains while
  drawing, track editor, REINFORCE and PPO written separately, no reference.
  Widest surface area, but slow training and no way to tell an environment bug
  from an algorithm bug.
- **B. Lean (chosen).** Headless vectorized numpy simulation with a Gymnasium
  interface; `pyray` only draws. Separate viewer process. Policy-gradient
  methods built as a progression (REINFORCE, then A2C, then PPO) so each step is
  a small, visible change. SB3 as an answer key. No editor in v1.
- **C. SB3 first.** Fastest results, but the learning happens inside a library,
  which defeats the goal. Rejected.

## How it works

### Track

- Track file (JSON): list of centerline points (closed loop), road width, start
  index, display name.
- A closed Catmull-Rom spline through the points, sampled to a few hundred
  points. Offsetting each sample left and right by half the width gives inner and
  outer walls as line segments.
- Progress: a car's position projected onto the sampled centerline, as a lap
  fraction in [0, 1). Gives a smooth per-step reward, lap detection, and
  wrong-way detection. Track the car's last centerline index and search only
  nearby indices, so projection is cheap and cannot jump across the track where
  two parts lie close together.
- Ships with two tracks: `oval.json` (easy) and `twisty.json` (corners of
  varying sharpness).
- A procedural track generator (random smooth closed loop, no self-
  intersection, width check) for the generalization stage.

### Car

- Kinematic bicycle model, grippy. State per car: position, heading, speed,
  steering angle. Fixed timestep `dt = 1/60`, independent of rendering.
- Actions: two continuous values in [-1, 1]: steering and throttle (negative =
  brake). Same action space for every learner, so results compare fairly.
- Crash: any edge of the car's rectangle crosses a wall segment. Only check wall
  segments near the car's progress index.

### Sensors and observation

- A fan of rays from the car's nose (default 9 rays over roughly ±90°, max
  range a fixed length). Distance to the first wall hit, divided by max range.
- Plus speed (normalized), current steering, and angle between car heading and
  the centerline direction.
- Ray-vs-segment intersection is one vectorized numpy operation over all cars,
  rays and nearby segments.

### Environment

- A batched simulator steps N cars at once with numpy arrays (the core,
  used by evolution and the from-scratch policy-gradient code).
- A thin single-car wrapper exposes the standard Gymnasium `reset()` /
  `step()` interface, so SB3 can train on the exact same physics and reward.
- Episode ends on crash, wrong way past a threshold, timeout, or after a set
  number of laps.
- Reward (default): progress gained this step times a scale, minus a crash
  penalty, minus a tiny per-step time cost, plus a lap bonus. The "fastest lap"
  stage changes the weights and explains why.

### Learners (each its own heavily commented file)

1. **Neuroevolution** (numpy only): population of small MLPs, forward pass
   written by hand, fitness = progress (later lap time), keep the best, mutate,
   repeat. No gradients. One generation = one batched episode.
2. **REINFORCE** (PyTorch): Gaussian policy, Monte Carlo returns.
3. **A2C**: REINFORCE plus a learned value function (critic) as a baseline.
4. **PPO**: A2C plus clipped objective, GAE, minibatch epochs.

Files 2 to 4 are separate so the owner can diff them; each lesson walks through
its diff from the previous one.

### Viewer and game modes

- `drive`: the owner drives with the keyboard.
- `train <method>`: headless, full speed. Writes to `runs/<method>-<timestamp>/`:
  checkpoints, a metrics CSV, the config used, and the best-lap trajectory per
  generation or iteration.
- `watch <run>`: a separate window. Every few seconds it loads the newest
  checkpoint and runs it live. Shows ghost cars for earlier best laps, a live
  neural-network panel (input values, hidden activations, outputs), a learning
  curve, and the lap time.
- `D` key toggles the debug overlay: walls, centerline, sensor rays, progress,
  stats.
- `compare`: plots learning curves and best lap times for all runs.
- `sb3`: trains SB3 PPO on the Gymnasium wrapper as the answer key.

### Learning material

- Audience: a senior CS student. Strong Python, comfortable with calculus,
  linear algebra and probability. New to neural networks and RL. So: no math
  primer; derive the policy-gradient theorem, advantage/baseline, GAE and the
  PPO clip objective properly but concisely; skip explaining Python, numpy
  basics or matrix multiplication; spend the words on RL intuition and on why
  each design choice exists.
- `docs/lessons/NN-<topic>.md`, one per stage, written for someone reading
  finished code: what the stage does, why, which lines matter (linked),
  what to look for in the viewer, and 2 to 4 experiments ("change X in the
  config, predict, run, compare").
- Settings live in config files so experiments need no code changes.
- Stage order: overview and how to run; track and geometry; car physics;
  sensors and the environment interface; neural nets and neuroevolution;
  REINFORCE; A2C; PPO; fastest lap and reward shaping; generalizing to random
  tracks; comparison with SB3 and final results.

### Quality

- pytest tests for geometry (ray intersection, segment crossing), progress and
  lap detection, physics sanity, and the Gymnasium interface (with
  Gymnasium's env checker).
- Each stage is verified by a real training run before its lesson is written.

## Success looks like

- The owner can drive the car on both tracks.
- Every method completes clean laps on `oval.json`. On `twisty.json`,
  neuroevolution, A2C and PPO complete clean laps. Plain REINFORCE may fail
  there; if it does, the lesson reports that honestly as the reason A2C exists,
  instead of adding tricks that blur the REINFORCE to A2C diff. (Claude's call.)
- Our PPO completes laps on `twisty.json` in at least 2 of 3 seeds, with a best
  lap within 10% of SB3 PPO's under the same step budget. (Claude's call.)
- Generalization stage: PPO only, trained on random tracks, evaluated on a fixed
  held-out set of random tracks plus `twisty.json`. Done when it completes laps
  on held-out tracks; partial failure is an acceptable, reported finding.
  (Claude's call.)
- Lap times visibly improve in the viewer and in the comparison plot.
- The owner can point at any behavior in the viewer and find the code that
  causes it.

## Runs, budgets and reproducibility (Claude's calls)

- Time budget on this machine: a main training run takes at most about 15
  minutes; a lesson experiment at most about 5 minutes. Parallel car count,
  network size and step budgets are tuned to fit.
- Every method gets the same environment-step budget per track, fixed in
  config before any main runs. The comparison plots learning curves against
  both environment steps and wall-clock time, and the final lesson discusses
  why the ranking can differ between the two.
- Runs are seeded but not bit-identical (multithreaded torch is allowed).
  Main results use 3 seeds and lessons report ranges, so the owner knows small
  differences on a rerun are normal.
- Every experiment in a lesson ships with Claude's measured result (1 seed),
  hidden in a collapsible "what we got" block below the prediction prompt.

## Left to the builder

Exact network sizes, hyperparameters, ray count and spread, reward constants,
episode limits, config file format, module layout, viewer layout details,
colors, and the order of work within each stage.
