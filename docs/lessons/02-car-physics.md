# 02 · Car physics

The car is a *kinematic bicycle* with a grip limit. It is deliberately simple: no tyre slip, no
drifting, no weight transfer. What it keeps is the one property that makes racing a real
control problem: **a car cannot take a corner faster than its tyres allow, so it has to slow
down before the corner, not in it.** That single rule is what the learners spend most of their
training discovering.

Files: [`racer/car.py`](../../racer/car.py), [`racer/config.py`](../../racer/config.py)
(`CarParams`), [`configs/env.toml`](../../configs/env.toml) (`[car]` table),
[`tests/test_car.py`](../../tests/test_car.py).

## State and actions

Each car is four numbers ([`CarState`](../../racer/car.py#L27-L42)): position (x, y) of its
center, heading (radians counter-clockwise from +x), forward speed (never negative, there is no
reverse) and the current front-wheel angle. N cars are four numpy arrays with one entry per car,
so one call to [`step_cars`](../../racer/car.py#L45-L75) moves every car at once.

The action is two numbers in [-1, 1], the same for every learning method: `steer` (+1 is full
left) and `throttle` (negative means brake). Physics runs at a fixed `dt = 1/60` s, whatever the
screen is doing; drive mode spends real frame time in whole physics steps through an accumulator
([drive.py#L68-L89](../../racer/drive.py#L68-L89)) so the car behaves identically on screen and in
training.

## One physics step, in order

All default values below are from [`configs/env.toml`](../../configs/env.toml). The "measured"
numbers come from [`docs/lessons/scripts/physics_numbers.py`](scripts/physics_numbers.py), which
steps the real `step_cars`.

### 1. Steering moves at a limited rate

```python
target = steer_cmd * max_steer                      # max_steer = 0.6 rad (34 degrees)
steer  = steer + clip(target - steer, -steer_rate * dt, +steer_rate * dt)   # steer_rate = 3 rad/s
```

[Lines 54-57.](../../racer/car.py#L54-L57) The wheel moves toward the commanded angle at no more
than 3 rad/s, so straight to full lock takes 12 steps = 0.200 s (measured). Without this limit a
policy could flip from full left to full right between two decisions, which no car can do and
which learners happily exploit (it looks like a vibrating car). It also means the wheel angle is
*state*: what the car does next depends on where the wheel is now, not only on the action. That
is why the observation includes it (lesson 03).
[`test_steer_rate_limited`](../../tests/test_car.py#L58-L60) pins it.

### 2. Speed: engine, brake, drag

```python
accel = max_accel * throttle   if throttle > 0   else   max_brake * throttle   # 8 and 16 m/s^2
accel = accel - drag * speed**2                                               # drag = 0.0015 /m
speed = clip(speed + accel * dt, 0, max_speed)                                # max_speed = 50 m/s
```

[Lines 59-62.](../../racer/car.py#L59-L62) Braking is twice as strong as the engine, like a real
car. Quadratic drag makes acceleration fade at speed. Measured: full throttle takes the car from
rest to the 50 m/s cap in 7.65 s and 211 m. Constant throttle 0.4 reaches 15.4 m/s after 5 s,
27.7 m/s after 10 s and levels off at 46.2 m/s (after 60 s), where drag cancels the engine.

### 3. The bicycle: turning from the wheel angle

Squash the car into one front wheel and one rear wheel, `wheelbase` (2.7 m) apart. If neither
wheel slides sideways, each wheel rolls along its own heading, so the car rotates about the point
where the two wheels' axle lines meet. Seen from above, car pointing up, wheel turned right by
`delta`:

```
   front wheel, turned right by delta
        /
       o .
       |    .          front axle line: perpendicular to the front wheel,
       |       .       so it leaves at angle delta below the horizontal
  L    |          .
       |             .
       o----------------C      rear axle line: perpendicular to the rear wheel
   rear wheel       R           C is the center of the turn
```

The triangle has a right angle at the rear wheel and angle `delta` at `C`, so
`tan(delta) = L / R`, or `R = wheelbase / tan(delta)`. Driving at speed `v` around a circle of
radius `R` turns the heading at `v / R` radians per second. Writing curvature `kappa = 1 / R`:

```
kappa      = tan(delta) / wheelbase
heading'   = speed * kappa
```

[Lines 64-68.](../../racer/car.py#L64-L68) At full lock the tightest circle is
`2.7 / tan(0.6) = 3.9 m` (measured: 3.9 m at 5 m/s;
[`test_low_speed_turns_tight`](../../tests/test_car.py#L84-L91) pins it). One simplification
([module docstring](../../racer/car.py#L7-L11)): the formula is for the rear axle, but `pos` is
the car's center. At these speeds and this wheelbase the difference is small, and it keeps every
other piece of geometry (corners, rays, rendering) centered on one point.

### 4. The grip limit

Moving on a circle of radius `R` at speed `v` needs a sideways (centripetal) acceleration of

```
a_lat = v^2 / R = v^2 * kappa
```

and tyres can only supply so much. The model caps curvature so that `a_lat` never exceeds
`max_lat_accel` (12 m/s², a bit over 1 g):

```python
grip_curvature = max_lat_accel / speed**2
kappa = clip(kappa, -grip_curvature, +grip_curvature)
```

When the cap binds, the car simply turns less than the wheel asks for: it *understeers* and runs
wide, instead of sliding or spinning. Measured turning radius at full lock: 3.9 m at 5 m/s,
33.3 m at 20 m/s, 75.0 m at 30 m/s. Steering harder does nothing past the limit; only slowing
down tightens the turn. [`test_grip_limit_caps_lateral_accel`](../../tests/test_car.py#L68-L81)
drives a car at 40 m/s with full lock and checks `speed * yaw_rate` never exceeds the limit (and
that the limit really binds).

### 5. Integrate

[Lines 70-73](../../racer/car.py#L70-L73) update the position with the *new* heading and speed
(semi-implicit Euler). It is as cheap as plain Euler and a little more stable.

The car's rectangle (4.5 m × 2 m) comes from [`car_corners`](../../racer/car.py#L78-L90) and
[`car_edges`](../../racer/car.py#L93-L100): four corners rotated by the heading. Those four edges
are what the crash test in lesson 01 checks against the walls.

## Why the grip limit creates the braking problem

Solve `a_lat = v² / R` for speed: the fastest a car can go round a corner of radius `R` is

```
v_max = sqrt(max_lat_accel * R)
```

The oval's end turns have their control points on 50 m circles, so `v_max = sqrt(12 × 50) =
24.5 m/s`. Now hold a gentle throttle of 0.4 down the first straight. The car is still
accelerating when it reaches the turn: 27.8 m/s at the start of the turn, 31.0 m/s at its
fastest. At 31.0 m/s the tightest curve the grip allows has radius `31.0² / 12 = 80.3 m`. The
corner is 50 m. The car runs wide and hits the outer wall 11.75 s after the start, 5.52 m right of
the centerline (the wall is 7 m out; the car's corner reaches it first). The same steering with a
throttle that coasts above 20 m/s laps cleanly. Experiment 2 reproduces both.

The fix is cheap if done early: braking from 31 m/s to 24.5 m/s takes 10.6 m (measured). The
difficulty is entirely *when*. For a learner this is hard in a specific way:

- Going faster pays more progress reward **right now**. The crash it causes arrives seconds
  later, after the moment when braking would have helped. Connecting the late penalty to the
  early decision is the credit assignment problem that every method in lessons 04-07 solves
  differently.
- The car cannot "see" corner radii. It has to learn that "forward ray getting short while speed
  is high" means "brake". The forward rays reach 60 m, so the information is there in time, but
  only if speed is in the observation too. It is.
- The quickest lap uses all of it: full throttle on straights, brake late and hard, the widest
  possible line through the corner (a wider line is a bigger `R`, which allows more speed).

With a higher grip limit the problem shrinks: corners allow more speed, and less braking is
needed. Experiment 1 measures how much.

## What to look for in the viewer

Drive it yourself: `uv run racer drive --track oval` (Up/W throttle, Down/S brake, Left/A and
Right steer, R restart, D overlay). Things to notice:

- Hold full throttle into the first turn and full steer: the car still drifts outward. That is
  the curvature clamp, not a bug.
- Brake before the turn instead and the same steering takes you round.
- The steering wheel needs a fraction of a second to swing across (0.2 s from straight to full
  lock); quick left-right taps barely move the car.

Note: `racer drive` has no override flag; to change car parameters for driving, edit the
`[car]` table in [`configs/env.toml`](../../configs/env.toml) (and change it back). Training
commands take `--env-set car.max_lat_accel=30` instead.

## Experiments

### 1. Grip limit: max_lat_accel 12 vs 30

**Change.** In [`configs/env.toml`](../../configs/env.toml) set `max_lat_accel = 30.0` under
`[car]`, drive the oval with `uv run racer drive --track oval`, and note your best lap. Then
set it back to 12.0 and compare.

Claude cannot use a keyboard, so the measured result below comes from a scripted driver instead:
[`docs/lessons/scripts/grip_laps.py`](scripts/grip_laps.py). It uses the P-controller from
[`tests/helpers.py`](../../tests/helpers.py#L56-L82) to steer (from the observation only,
like a learned policy) and holds one target speed the whole lap. For each grip limit it tries
target speeds from 10 to 46 m/s and keeps the fastest lap that finishes without crashing. It
does the same on twisty.

**Predict.** Using `v_max = sqrt(max_lat_accel × R)` with `R = 50 m`: what is the fastest
constant speed that survives the oval at grip 12, and at grip 30? Roughly how much faster is the
best lap? On twisty, whose tightest corner is about 12 m, will a constant-speed driver be quick?

**Run.**

```sh
uv run python docs/lessons/scripts/grip_laps.py      # about 30 s
```

<details><summary>What we got</summary>

Output of `docs/lessons/scripts/grip_laps.py`, run 2026-11-23. "crash" = hit a wall; "slow" =
the 90 s time limit came before one lap.

```
oval   max_lat_accel=  12  best lap 30.28 s at target 24 m/s (2 laps)
    target:lap  10:72.82  12:60.95  14:52.52  16:46.22  18:40.22  20:36.22  22:32.93  24:30.28  26:crash  28:crash  30:crash  32:crash  34:crash  36:crash  38:crash  40:crash  42:crash  44:crash  46:crash
oval   max_lat_accel=  30  best lap 18.65 s at target 40 m/s (2 laps)
    target:lap  10:72.82  12:60.95  14:52.52  16:46.22  18:40.22  20:36.22  22:32.93  24:30.20  26:27.92  28:25.92  30:24.22  32:22.72  34:21.38  36:20.22  38:19.22  40:18.65  42:crash  44:crash  46:crash
twisty max_lat_accel=  12  best lap 82.52 s at target 14 m/s (1 laps)
    target:lap  10:slow  12:slow  14:82.52  16:crash  18:crash  20:crash  22:crash  24:crash  26:crash  28:crash  30:crash  32:crash  34:crash  36:crash  38:crash  40:crash  42:crash  44:crash  46:crash
twisty max_lat_accel=  30  best lap 53.47 s at target 22 m/s (1 laps)
    target:lap  10:slow  12:slow  14:82.35  16:72.35  18:64.58  20:58.42  22:53.47  24:crash  26:crash  28:crash  30:crash  32:crash  34:crash  36:crash  38:crash  40:crash  42:crash  44:crash  46:crash
```

- **Oval:** the highest surviving target is 24 m/s at grip 12 and 40 m/s at grip 30; the formula
  gives 24.5 and `sqrt(30 × 50) = 38.7` m/s. (The car can beat 38.7 slightly by using the 14 m
  road width for a wider line.) Best lap drops from 30.28 s to 18.65 s, 38% faster. Below 24 m/s
  the grip limit never binds, so both settings give the same lap times.
- **Twisty:** a constant-speed driver is slow. At grip 12 only 14 m/s survives (82.52 s; at 10
  and 12 m/s the lap takes longer than the 90 s limit), because the whole lap has to be driven at
  the speed of the tightest corner. Grip 30 raises that to 22 m/s (53.47 s).
- For scale: the 500,000-step evolution run from lesson 00
  (`runs/evolution-20261123-193221-s0`, grip 12) already laps the oval in 28.45 s, faster than
  any constant speed here. It changes speed along the lap.

The twisty numbers are the braking problem in one line: the only way to be fast there is to
drive fast on the straights and slow in the corners, which a fixed-speed rule cannot do and a
learned policy can.

</details>

### 2. Constant throttle 0.4 on the oval

**Change.** Nothing to edit. [`docs/lessons/scripts/constant_throttle.py`](scripts/constant_throttle.py)
runs the helper's P-controller twice on the oval: as written (throttle 0.4, coasting above
20 m/s) and with the throttle stuck at 0.4.

**Predict.** How fast is the car when it reaches the first turn with constant 0.4? Will it make
the turn? If not, which wall does it hit, the inner or the outer one, and why?

**Run.**

```sh
uv run python docs/lessons/scripts/constant_throttle.py
```

<details><summary>What we got</summary>

Output of `docs/lessons/scripts/constant_throttle.py`, run 2026-11-23:

```
oval: width 14.0 m (walls 7.0 m either side), corner radius 50.0 m
grip limit 12.0 m/s^2 -> fastest speed round a 50.0 m corner: sqrt(12.0 * 50.0) = 24.5 m/s

coast above 20 m/s:
  ended: finished (truncated) after 74.93 s, laps 2 (39.07 s, 35.82 s)
  speed at the end of the first straight (x = 100 m): 20.0 m/s, top speed 20.2 m/s
  final position [-48.8 -50.3], -0.31 m from the centerline
  at 20.2 m/s the tightest turn the grip allows has radius v^2 / a = 33.9 m

constant 0.4:
  ended: crashed after 11.75 s, laps 0 (none)
  speed at the end of the first straight (x = 100 m): 27.8 m/s, top speed 31.0 m/s
  final position [145.8 -31.4], -5.52 m from the centerline
  at 31.0 m/s the tightest turn the grip allows has radius v^2 / a = 80.3 m
```

The constant-throttle car enters the turn at 27.8 m/s, keeps accelerating to 31.0 m/s, and can
then turn no tighter than an 80 m radius against a 50 m corner. It drifts to the **outside**
(negative offset = right of the centerline, which is the outer side of a counter-clockwise
turn) and crashes at x = 145.8 m, a third of the way round the turn. The coasting car keeps
20 m/s, well under the 24.5 m/s limit, and drives both laps (the episode ends by `max_laps`,
which is a truncation, lesson 03).

</details>

## Tests worth reading

| test | what it pins |
|---|---|
| [`test_grip_limit_caps_lateral_accel`](../../tests/test_car.py#L68-L81) | lateral acceleration never exceeds `max_lat_accel`, and the limit binds |
| [`test_low_speed_turns_tight`](../../tests/test_car.py#L84-L91) | at low speed the radius is `wheelbase / tan(max_steer)` |
| [`test_steer_rate_limited`](../../tests/test_car.py#L58-L60) | the wheel moves at most `steer_rate * dt` per step |
| [`test_brake_never_reverses`](../../tests/test_car.py#L48-L55) | braking stops at zero speed |
| [`test_lap_counted_once`](../../tests/test_sim.py#L69-L101) | the coasting P-controller laps the oval (its comment explains the 20 m/s coast) |

Next: [03 · Sensors and the environment](03-sensors-and-env.md).
