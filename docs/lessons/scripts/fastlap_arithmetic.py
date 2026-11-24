"""Lesson 08: the reward arithmetic of a 41 s and a 45 s driver, by hand, from the config files.

A deliberately simple model: a driver who laps twisty in exactly T seconds at constant speed
(no standing start, no crash). It prints

  1. the undiscounted episode return (what evolution ranks) under three env configs: env.toml,
     the first env_fastlap.toml (max_laps 3, 90 s limit) and the fixed env_fastlap.toml;
  2. what discounting at gamma = 0.99 does: how much a lap bonus T seconds away is still worth,
     and the discounted value of driving on at constant speed, which is what a critic in A2C or
     PPO estimates (those methods bootstrap through truncation, so the episode never "ends"
     for them except by a crash or wrong way);
  3. the time cost seen from a bootstrapping method: a constant plus a bonus of
     time_cost / (1 - gamma) on every termination, and so the speed below which ending the
     episode beats driving on, for each time cost of lesson 08's experiment.

Run: uv run python docs/lessons/scripts/fastlap_arithmetic.py   (instant)
"""

import math

from racer.config import load_env_config
from racer.track import load_track

GAMMA = 0.99  # configs/ppo.toml and configs/a2c.toml
LAP_TIMES = (41.0, 45.0)


def episode_return(cfg, lap_s: float, length: float) -> tuple[float, str]:
    """Undiscounted return of a constant-speed driver lapping in ``lap_s`` seconds."""
    decisions_per_s = 1.0 / (cfg.dt * cfg.action_repeat)
    finish_s = cfg.max_laps * lap_s
    if finish_s <= cfg.max_episode_s:            # ends at the line after max_laps laps
        seconds, meters, laps, how = finish_s, cfg.max_laps * length, cfg.max_laps, "laps"
    else:                                        # the time limit cuts it off first
        seconds, how = cfg.max_episode_s, "time limit"
        meters = length * seconds / lap_s
        laps = math.floor(seconds / lap_s)
    ret = (cfg.progress_scale * meters + cfg.lap_bonus * laps
           - cfg.time_cost * decisions_per_s * seconds)
    return ret, f"{how} at {seconds:5.1f} s, {meters:6.1f} m, {laps} laps"


def main() -> None:
    length = load_track("twisty").length
    configs = {
        "env.toml": load_env_config("configs/env.toml"),
        "old env_fastlap": load_env_config("configs/env.toml", {"time_cost": 0.05, "max_laps": 3}),
        "env_fastlap.toml": load_env_config("configs/env_fastlap.toml"),
    }
    base = configs["env.toml"]
    per_s = 1.0 / (base.dt * base.action_repeat)  # 15 decisions per second
    print(f"twisty {length:.2f} m per lap; {per_s:.0f} decisions per second\n")

    print("1. undiscounted episode return (evolution's fitness)")
    for name, cfg in configs.items():
        rets = []
        for t in LAP_TIMES:
            ret, how = episode_return(cfg, t, length)
            rets.append(ret)
            print(f"  {name:17} {t:.0f} s laps: {how}  return {ret:7.2f}")
        print(f"  {name:17} 41 s driver earns {rets[0] - rets[1]:+.2f} more\n")

    print(f"2. discounting at gamma = {GAMMA}")
    per_second = GAMMA ** per_s
    print(f"  one second = {per_s:.0f} decisions: a reward 1 s away counts {per_second:.3f}; "
          f"half-life {math.log(0.5) / math.log(per_second):.1f} s; "
          f"effective horizon 1/(1-gamma) = {1 / (1 - GAMMA):.0f} decisions = "
          f"{1 / (1 - GAMMA) / per_s:.2f} s")
    for t in LAP_TIMES:
        bonus = base.lap_bonus * GAMMA ** (per_s * t)
        speed = length / t
        drive = base.progress_scale * speed / per_s / (1 - GAMMA)
        print(f"  {t:.0f} s laps ({speed:.2f} m/s): the first lap bonus, {t:.0f} s away, is worth "
              f"{bonus:.4f} at the start; driving on forever is worth "
              f"{drive:.2f} in progress (0.1 x v / 15 per decision, / (1 - gamma))")
    print()

    print("3. the time cost for a method that bootstraps truncation")
    print(f"  sum_t gamma^t (r_t - c) = sum_t gamma^t r_t - c/(1-gamma) + gamma^T c/(1-gamma) "
          f"for an ending at step T: a bonus of c/(1-gamma) on every termination")
    for mult in (1, 2, 5, 20):
        c = base.time_cost * mult
        bonus = c / (1 - GAMMA)
        # Driving on at speed v is worth 0.1 v / 15 / (1 - gamma) - c / (1 - gamma) (ignoring
        # lap bonuses, worth < 0.03 from more than 41 s away). Crashing now is worth
        # -crash_penalty; turning round (wrong way) about -0.1 x 15 m = -1.5.
        crash_v = max(0.0, (c / (1 - GAMMA) - base.crash_penalty)
                      * (1 - GAMMA) * per_s / base.progress_scale)
        wrong_v = max(0.0, (c / (1 - GAMMA) - base.progress_scale * base.wrong_way_m)
                      * (1 - GAMMA) * per_s / base.progress_scale)
        print(f"  time cost x{mult:<2} (c = {c:.2f}): termination bonus {bonus:5.1f}; ending beats "
              f"driving on below {crash_v:4.1f} m/s by crashing, {wrong_v:4.1f} m/s by turning round")


if __name__ == "__main__":
    main()
