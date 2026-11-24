# 05 · REINFORCE: the policy gradient

Evolution (lesson 04) treated a whole episode as one number and learned only from comparing
those numbers. This lesson trains the same kind of network by gradient descent, and gets a
learning signal from every single decision. The trick that makes it possible: make the policy
*random*, and differentiate the probabilities of the actions it took instead of the simulator.
The result, REINFORCE (Williams, 1992), is the plainest policy-gradient method there is. It is
written here with no improvements at all, so its weaknesses are on display; lessons 06 and 07
fix them one at a time.

Files: [`racer/algos/reinforce.py`](../../racer/algos/reinforce.py),
[`racer/nets_torch.py`](../../racer/nets_torch.py), [`configs/reinforce.toml`](../../configs/reinforce.toml),
[`configs/budgets.md`](../../configs/budgets.md), [`tests/test_reinforce.py`](../../tests/test_reinforce.py),
[`tests/test_nets_torch.py`](../../tests/test_nets_torch.py).

## A policy that rolls dice

Evolution's network mapped an observation to one action. REINFORCE's network maps it to a
*probability distribution* over actions, and the car drives by sampling from it.
[`GaussianActor`](../../racer/nets_torch.py#L30-L48) is the distribution:

```python
self.net = _mlp([obs_dim, *hidden, act_dim])                         # 12-64-64-2, tanh between layers
self.log_std = nn.Parameter(torch.full((act_dim,), LOG_STD_INIT))   # one number per action

def dist(self, obs):
    mean = self.mean(obs)                                            # = self.net(obs)
    return Normal(mean, self.log_std.exp().expand_as(mean))
```

For each observation `s`, the network outputs a mean `μ_θ(s) = (μ_steer, μ_throttle)`, and each
action component is drawn independently from a normal distribution around it with standard
deviation `σ_k = exp(log_std_k)`. The std is a free parameter, the same in every state, learned
along with the weights; it starts at `exp(-0.5) = 0.61` ([`LOG_STD_INIT`](../../racer/nets_torch.py#L17)).
`θ` stands for all 5,124 learnable numbers: the weights and biases of the 12-64-64-2 network
(`12·64+64 + 64·64+64 + 64·2+2 = 5,122`) plus the two `log_std`s.

Two differences from lesson 04's network matter later:

- **No tanh on the output.** `_mlp` puts tanh between layers but not after the last one
  ([nets_torch.py#L20-L27](../../racer/nets_torch.py#L20-L27)), so the mean can be any real
  number. Actions must be in [-1, 1]; the simulator clips them ([reinforce.py#L103](../../racer/algos/reinforce.py#L103)
  says so). Lesson 06 shows what an unbounded mean does over a long run.
- **Training and driving use different actions.** While training, the car drives the *sample*
  (that is the exploration). For evaluation and in the viewer it drives the mean, clipped to
  [-1, 1] ([`TorchPolicy.act`](../../racer/algos/reinforce.py#L58-L61)). Those are not the same
  driver, and this lesson's results turn on the difference.

### The log-probability

The density of a one-dimensional normal is `N(a; μ, σ) = exp(-(a - μ)² / (2σ²)) / (σ √(2π))`.
The two components are independent, so the density of the action pair is the product, and its
logarithm is a sum:

```
log π_θ(a | s) = Σ_k [ -(a_k - μ_k(s))² / (2 σ_k²)  -  log σ_k  -  ½ log 2π ]      k ∈ {steer, throttle}
```

That is [line 170](../../racer/algos/reinforce.py#L170): `actor.dist(batch_obs).log_prob(batch_act).sum(-1)`,
torch's per-component log density summed over the two actions.
[`test_logprob_matches_distribution`](../../tests/test_nets_torch.py#L16-L22) checks the sum
against two separate one-dimensional normals. Its derivatives are what the learning rule uses:

```
∂ log π / ∂μ_k        = (a_k - μ_k) / σ_k²            pull the mean towards the sampled action
∂ log π / ∂log σ_k    = (a_k - μ_k)² / σ_k² - 1        widen if the sample was far out, narrow if close
```

(the first one then flows back through the network into every weight, by backpropagation).

**Which action does the log-probability use?** The *unclipped* sample `a`. If the network says
`μ_throttle = 0.9` and the sample is 1.4, the car gets throttle 1.0, but the log-probability is
computed for 1.4 ([nets_torch.py#L33-L36](../../racer/nets_torch.py#L33-L36)). This is exact,
not an approximation: the policy's action *is* the sample, and clipping is just part of how the
environment responds to it, like the grip limit. The policy-gradient derivation below never asks
the environment to be differentiable, so a clip inside it costs nothing. (The alternative,
computing the probability of the clipped value 1.0, would be wrong: a continuous density does not
give the probability that the action "was 1.0".)

## The objective

Write a whole episode as a trajectory `τ = (s_0, a_0, r_0, s_1, a_1, r_1, ..., s_{T-1}, a_{T-1}, r_{T-1})`.
How likely a trajectory is depends on the start state, the policy and the simulator:

```
P_θ(τ) = p(s_0) · Π_t π_θ(a_t | s_t) · p(s_{t+1} | s_t, a_t)
```

The simulator `p(s_{t+1} | s_t, a_t)` is deterministic here (every episode starts at the same
spot, and the physics have no noise), which is fine: it is the special case where all the
probability sits on one next state. All the randomness in `τ` comes from the policy's dice.

The goal is to make the expected return large:

```
J(θ) = E_{τ ~ P_θ} [ R(τ) ],      R(τ) = Σ_t γ^t r_t
```

This sidesteps lesson 04's problem. There, `F(θ)` was the return of *one* deterministic
episode, a function of `θ` full of jumps (a crash is on or off). `J(θ)` is an *average* over
episodes, weighted by probabilities that depend smoothly on `θ`: nudge a weight and every
trajectory becomes a little more or a little less likely, so `J` changes smoothly even though
each single return jumps.

## Deriving the policy gradient

Write the expectation as an integral (a sum, if you prefer to think of finitely many
trajectories) and differentiate:

```
∇_θ J = ∇_θ ∫ P_θ(τ) R(τ) dτ = ∫ ∇_θ P_θ(τ) R(τ) dτ
```

`R(τ)` does not depend on `θ` once the trajectory is fixed; only how likely it is does. Now the
**log-derivative trick**: for any positive function, `∇ log f = ∇f / f`, so `∇f = f ∇ log f`.
Applied to `P_θ`:

```
∇_θ J = ∫ P_θ(τ) ∇_θ log P_θ(τ) R(τ) dτ = E_{τ ~ P_θ} [ ∇_θ log P_θ(τ) · R(τ) ]
```

The gradient is again an expectation over trajectories from the current policy, so it can be
estimated by driving episodes and averaging. What is `∇ log P_θ(τ)`? The log of the product is a
sum:

```
log P_θ(τ) = log p(s_0) + Σ_t log π_θ(a_t | s_t) + Σ_t log p(s_{t+1} | s_t, a_t)
```

The first and last terms are the simulator. They do not contain `θ`, so their gradient is zero:

```
∇_θ log P_θ(τ) = Σ_t ∇_θ log π_θ(a_t | s_t)
```

and therefore

```
∇_θ J = E_τ [ Σ_t ∇_θ log π_θ(a_t | s_t) · R(τ) ]                                  (1)
```

This is the whole idea. **We never differentiate the simulator.** We only need the gradient of
the policy's own log-probability of the actions it took, which torch computes by
backpropagation through our small network. The simulator only has to be *run*. That is the
answer to lesson 04's "there is no useful derivative to take": we take a derivative of something
else whose expectation is the one we want.

`∇_θ log π_θ(a_t | s_t)` is called the *score*. Equation (1) says: for each episode, push up the
log-probability of every action it took, in proportion to how good the episode was. Episodes with
high return become more likely, ones with low return less likely.

### Reward-to-go: only the future counts

Equation (1) credits action `a_t` with the whole episode's return, including rewards collected
*before* `t`, which `a_t` could not have caused. Those terms average to zero. Take a reward `r_k`
from before the action, `k < t`. Condition on everything that happened up to `s_t` (call it
`h_t`); then `r_k` is a fixed number, and only `a_t ~ π_θ(· | s_t)` is random:

```
E[ r_k ∇ log π_θ(a_t | s_t) | h_t ] = r_k · E_{a ~ π_θ(·|s_t)} [ ∇ log π_θ(a | s_t) ]
                                    = r_k · ∫ π_θ(a | s_t) · ∇π_θ(a | s_t) / π_θ(a | s_t) da
                                    = r_k · ∇ ∫ π_θ(a | s_t) da  =  r_k · ∇ 1  =  0
```

The expected score is always zero, because the probabilities always integrate to 1, whatever
`θ` is. (Remember this line: lesson 06 uses exactly the same argument to prove that a baseline
does not bias the gradient.) So drop the past rewards from each step's weight and keep only the
*reward-to-go*:

```
∇_θ J ≈ E_τ [ Σ_t ∇_θ log π_θ(a_t | s_t) · G_t ],     G_t = Σ_{k ≥ t} γ^{k-t} r_k      (2)
```

Same expectation (with one caveat, next paragraph), fewer random terms, so less variance: each
step's weight now only contains the noise of what came after it.

The caveat. Taking (1) literally with `R(τ) = Σ γ^k r_k`, the weight of step `t` would be
`Σ_{k≥t} γ^k r_k = γ^t G_t`: late steps would count less, because the objective discounts
everything from the start of the episode. Practically every implementation, this one included,
drops that `γ^t` and uses `G_t`. Then `γ` is no longer "how much the objective cares about the
far future" but "how far ahead a decision's consequences are credited to it". Strictly, (2) is
then not the exact gradient of any objective (a known, accepted bias; Nota and Thomas,
"Is the Policy Gradient a Gradient?", 2020); what it does is
treat every step's decision as equally important, which is what we want for a car that should
drive well at second 80 as much as at second 1. With `γ = 0.99`, rewards 100 decisions (6.7 s)
ahead count `0.99^100 = 0.37`, and 300 ahead (20 s) count 0.05 (lesson 03 has the arithmetic).

### From formula to code

Estimate the expectation in (2) by averaging over a batch of sampled steps, and turn it into a
loss whose gradient is that estimate:

```
loss(θ) = -(1/N) Σ_{all steps i in the batch} log π_θ(a_i | s_i) · G_i
```

`G_i` is a plain number (computed from the rewards, not from `θ`), so `∇ loss` is exactly minus
the estimate of (2), and one gradient-descent step on `loss` is one gradient-ascent step on `J`.
That is [`reinforce_loss`](../../racer/algos/reinforce.py#L131-L133):

```python
return -(logp * returns).mean()
```

This loss is a *surrogate*: its value means nothing (it is not "how bad the policy is", and it
can go up while the policy improves). Only its gradient matters. Dividing by the number of steps
`N` rather than the number of episodes only rescales the gradient, and Adam's update is
insensitive to a constant rescaling.

[`test_reinforce_gradient_direction`](../../tests/test_reinforce.py#L17-L33) checks the sign on
the smallest possible problem, a one-step bandit with reward `-(a - 0.5)²`: 200 steps of this loss
move the mean from below 0.3 to within 0.1 of 0.5.

## One iteration, line by line

[`train_reinforce`](../../racer/algos/reinforce.py#L139-L204) loops over four steps until the
budget is spent.

**1. Collect** ([`collect_batch`](../../racer/algos/reinforce.py#L81-L114)). Reset all
`n_cars = 16` cars to the start line, then drive them with sampled actions
([line 102](../../racer/algos/reinforce.py#L102)), keeping a growing list of observations, actions
and rewards per car. When a car's episode ends (crash, wrong way, time limit or 2 laps), its lists
become one finished episode and the auto-reset sim respawns it. Stop as soon as
`episodes_per_batch = 32` episodes have finished; the cars still mid-episode are thrown away.
Why reset first: without it, the first episode a car finished in this batch would be the tail
of one begun under the previous policy, starting mid-track, and its reward-to-go would be
computed from a start it did not have ([`test_every_episode_starts_at_a_spawn`](../../tests/test_reinforce.py#L55-L65)).

**2. Returns** ([`discounted_returns`](../../racer/algos/reinforce.py#L119-L126)), per episode:

```python
running = 0.0
for t in reversed(range(len(rewards))):
    running = rewards[t] + gamma * running      # G_t = r_t + γ G_{t+1}, with G_T = 0
    returns[t] = running
```

Computed backwards, it is one pass instead of a sum per step. Per episode, so a crash at the end
of one episode never leaks into the previous one's returns
([`test_discounted_returns`](../../tests/test_nets_torch.py#L11-L13): rewards `[1, 1, 1]` with
`γ = 0.5` give `[1.75, 1.5, 1.0]`).

**3. Loss** ([lines 166-171](../../racer/algos/reinforce.py#L166-L171)). The log-probabilities are
recomputed from the stored observations and actions, this time with gradients. The policy has not
changed since collection, so these are the same numbers the sampler would have reported.

**4. Update** ([lines 174-176](../../racer/algos/reinforce.py#L174-L176)). One Adam step
(`lr = 3e-4`) on the 5,124 parameters, then the batch is thrown away. It was sampled from the
policy *before* this step; equation (2) is an expectation under the *current* policy, so after one
step the old samples no longer estimate it. This is what *on-policy* means, and it is expensive:
each batch, tens of thousands of decisions, buys exactly one gradient step.

Every `eval_every = 5` iterations, and on the last, the mean policy drives one
[`evaluate`](../../racer/evaluate.py#L28-L57) episode and a checkpoint is saved
([lines 189-202](../../racer/algos/reinforce.py#L189-L202)), exactly as for evolution. The two
extra `metrics.csv` columns are the batch's mean episode length and the mean `log_std`
([line 36](../../racer/algos/reinforce.py#L36)).

## The first batch: every action looks bad

What does the gradient see at the very start? [`docs/lessons/scripts/reinforce_first_batch.py`](scripts/reinforce_first_batch.py)
rebuilds iteration 1 exactly as `racer train reinforce --set track=oval` does it (same seed, same
initial network, same `collect_batch`; its mean return and length, -14.45 and 869.1, are the
first row of seed 0's `metrics.csv` below) and sorts the 32 episodes by how they ended:

```
uv run python docs/lessons/scripts/reinforce_first_batch.py oval 0

oval, seed 0: first batch = 32 episodes, 36,608 car-decisions simulated (unfinished episodes dropped)
  crash     : 28 episodes, 22,410 of the batch's steps
      length  800.4 (min 455, max 1282), progress  27.6 m, return  -15.24, G_0  -0.66
      mean reward-to-go: over all its steps  -1.82, over its last 100 steps  -6.56
  wrong way : none
  time limit:  4 episodes, 5,400 of the batch's steps
      length 1350.0 (min 1350, max 1350), progress  46.0 m, return   -8.90, G_0  -0.55
      mean reward-to-go: over all its steps  -0.62, over its last 100 steps  -0.25
  max laps  : none
  reward-to-go over the whole batch: max 0.015, min -10.010, 27,802 of 27,810 steps negative
reference: sit still 1350 decisions -> G_0 = -1.000 (undiscounted -13.50); crash on the first decision -> G_0 = -10.01
```

(twisty, seed 0, looks the same: 28 crashes, 4 time limits.) The untrained network outputs means
near 0, so the car gets throttle around 0 plus noise of std 0.61: it creeps (27.6 m in 53 s on
average) and wobbles until it touches a wall. Two things follow.

- **All the weights are negative.** Progress is worth 0.1 per meter, and a creeping car makes
  almost none, so the reward-to-go is negative at practically every step (27,802 of the 27,810
  kept). Plain REINFORCE
  therefore pushes the probability of *every* sampled action down; it learns only from *how much*
  each is pushed. There is no "good action" anywhere in this batch.
- **Crashes are pushed hardest.** In the last 100 decisions before a crash the weight averages
  -6.56, because the -10 penalty is only discounted by `0.99^k` for a crash `k` steps ahead. A car
  that sits still for the whole 90 s costs at most 1.0 discounted (`0.01 · (1 - 0.99^1350) / 0.01`),
  and its weights are around -0.25. So the steps that lead to walls get pushed away from much
  harder than anything else, and the cheapest way to stop reaching a wall is to stop moving
  towards it.

The runs below show the result. In the first 15-20 iterations the mean episode length rises from
335-870 decisions to about 1,300 on every seed and track (`episode_len_mean` in the six
main runs' `metrics.csv`): the cars stop crashing by barely moving.
[`docs/lessons/scripts/stochastic_vs_mean.py`](scripts/stochastic_vs_mean.py) on oval seed 1's
checkpoints shows it:

```
uv run python docs/lessons/scripts/stochastic_vs_mean.py runs/reinforce-20261123-204930-s1 5 15 25

ckpt_00005.pt  mean policy : 0 laps, crashed after 54 m, ..., top speed 5.8 m/s
               noisy policy: 32 episodes, laps 0/1/2 = 32/0/0, crashed 26, ..., median top speed 4.4 m/s
ckpt_00015.pt  mean policy : 0 laps, no crash after 0 m, avg applied throttle -0.02, top speed 0.0 m/s
               noisy policy: 32 episodes, laps 0/1/2 = 32/0/0, crashed 8, ..., median top speed 3.3 m/s
ckpt_00025.pt  mean policy : 0 laps, no crash after 0 m, avg applied throttle -0.02, top speed 0.0 m/s
               noisy policy: 32 episodes, laps 0/1/2 = 32/0/0, crashed 6, ..., median top speed 3.4 m/s
```

(`crashed` there counts crash or wrong way.) Crashes among 32 noisy episodes fall from 26 to 6,
at a crawl of about 3 m/s. The *mean* policy goes further: its throttle on the start line is
-0.019 at iteration 15 ([`docs/lessons/scripts/policy_means.py`](scripts/policy_means.py)), a hair
below zero, and a negative throttle on a stationary car is a brake that holds it in place. So the
deterministic evaluation reports 0.0 m from iteration 10 to 25 (`eval_progress_m` in its
`metrics.csv`) while the noisy cars still creep about. "Learns to sit still" is literally true
only for the mean; the noisy policy learns to *crawl*. With a larger learning rate the push is
stronger: in Experiment 2's `lr = 3e-3` run on seed 1 (`runs/reinforce-20261123-212229-s1`) the
noisy policy's applied throttle averages -0.60 at iteration 50 (`stochastic_vs_mean.py ... 50`),
and the mean never moves the car in 4.5M steps.

Getting out of the crawl needs batches where some cars happen to move and *not* crash, so that
their less negative (later: positive) returns pull the policy back towards throttle. That takes
hundreds of thousands of steps, and every seed got out eventually on the oval.

## Variance: how much is one gradient worth?

Equation (2) is exact *in expectation*. One batch gives one sample of it, and the sample is noisy
for three reasons visible above: `G_t` sums hundreds of random rewards; the weights share a
large common offset (all negative), which only cancels on average; and the score
`(a - μ)/σ²` is itself random with mean zero. How noisy, concretely?
[`docs/lessons/scripts/gradient_noise.py`](scripts/gradient_noise.py) takes one checkpoint
(oval seed 0 at iteration 25, the first that lapped), collects 8 independent batches with it and
computes the gradient of each, without stepping:

```
uv run python docs/lessons/scripts/gradient_noise.py runs/reinforce-20261123-204929-s0 25 32 8

8 batches of 32 episodes (mean 43,200 decisions simulated per batch), 5,124 parameters
  cosine similarity between batch gradients: mean +0.396, min -0.296, max +0.909
  |mean gradient| / mean |gradient| = 0.678  (pure noise would give about 1/sqrt(8) = 0.354)

uv run python docs/lessons/scripts/gradient_noise.py runs/reinforce-20261123-204929-s0 25 8 8

8 batches of 8 episodes (mean 21,506 decisions simulated per batch), 5,124 parameters
  cosine similarity between batch gradients: mean +0.112, min -0.778, max +0.924
  |mean gradient| / mean |gradient| = 0.376
```

Two batches of 32 whole episodes (43,200 decisions each) from the *same* policy point in
directions with cosine similarity 0.40 on average, and some pairs point in partly opposite
directions (-0.30). With 8 episodes per batch the agreement drops to 0.11, barely above pure
noise: the second ratio, 0.376, is close to the 0.354 that zero-mean noise would give.

At this checkpoint nearly every car drives the full 90 s, so an 8-episode batch usually holds all
16 episodes, finishing on the same tick: 21,506 decisions on average, not a quarter of 43,200.

That is the price of the simple estimator: each update needs tens of thousands of decisions to
point roughly the right way, and each batch buys only one update. Lesson 06's critic attacks
exactly this.

## Two biases

### Truncation: the time limit treated as a crash

`discounted_returns` starts every episode's recursion at 0, whatever ended it. For a crash that is
right: nothing follows. For the 90 s time limit (or the end of the second lap) it is not: the car
was fine and would have kept scoring; the episode was cut off by a rule. Seen from the policy,
which does not know the time (it is not in the observation, lesson 03), the same state then has a
short future near the time limit and a long one earlier.

Some arithmetic. A car holding 20 m/s earns `0.1 × 20 × 4/60 - 0.01 = 0.123` per decision. If it
could go on forever, its discounted reward-to-go would be `0.123 / (1 - 0.99) = 12.3`. `k`
decisions before the cutoff it is `12.3 × (1 - 0.99^k)`: 0.12 on the last step, 6.2 with 69 steps
(4.6 s) to go, within 5% of 12.3 only with 300 or more to go. So in the last ~20 s of every
time-limited episode the same good driving is credited with less and less, down to almost nothing,
and those steps form a sizeable share of the data (up to 300 of 1,350 decisions per such episode).

Is this "wrong"? Strictly, plain Monte Carlo REINFORCE is unbiased for the objective it is given
(up to the dropped `γ^t` from the caveat above): the discounted return *of a 90 s episode*. The trouble is that a policy that cannot see the clock
cannot act on that objective anyway; what we want is a car that drives well everywhere, and
for that the time-limit cut is a bias in the weights: a perfectly good action at second 88 gets
a weight near 0, as if it had not mattered. Lesson 06 fixes it by asking a critic what the state
was worth and adding that instead of 0 (the *truncation bootstrap*).

### The batch rule favours short episodes

`collect_batch` stops at the first 32 *finished* episodes and drops the ones still running. A car
whose episode ends early (a crash) starts a new one and can contribute two or three; a car in a
long, good episode contributes at most one, and its unfinished successor is thrown away. So
short episodes are over-represented, and the gradient is taken over a batch with more crashes in
it than the policy actually produces. [`docs/lessons/scripts/batch_bias.py`](scripts/batch_bias.py)
measures it at a checkpoint where crashes and full episodes are mixed (oval seed 0, iteration 35):
the unbiased reference lets 32 cars drive one whole episode each, the batch rule is
`collect_batch` with 16 cars, both repeated over 10 seeds:

```
uv run python docs/lessons/scripts/batch_bias.py runs/reinforce-20261123-204929-s0 35 32 10

  one episode per car (unbiased)       mean length 1069.0, median   1350, crash or wrong way 42.8%, ...
  collect_batch, first 32 of 16 cars   mean length 1005.6, median   1350, crash or wrong way 47.6%, ...
  collect_batch simulated 41,968 decisions per batch and kept 34,192 (81%); the rest were dropped unfinished episodes
```

Crashes make up 47.6% of the batch against 42.8% of what the policy really does, and 19% of the
simulated decisions are paid for and thrown away. The bias is moderate here because time-limited
episodes all end on the same tick (every car starts together after the reset), so whole groups
finish at once. It is a cost of insisting on whole episodes; lesson 06 drops that requirement
and with it this bias.

## Results: three seeds on each track

Six main runs with the defaults ([`configs/reinforce.toml`](../../configs/reinforce.toml):
16 cars, 32 episodes per batch, `γ = 0.99`, `lr = 3e-4`, 4.5M env steps), seeds 0, 1, 2 on oval
(`--set track=oval`, same budget as every method, [`configs/budgets.md`](../../configs/budgets.md))
and on twisty, run six at a time on the dev machine, 2026-11-23. Their wall times are inflated:
for about 2.5 minutes six more training processes shared the machine by mistake, so this lesson
quotes env steps, not seconds.

```sh
uv run racer train reinforce --seed 0 --set track=oval     # and --seed 1, 2; and without --set for twisty
uv run python docs/lessons/scripts/noisy_vs_mean_laps.py runs/reinforce-20261123-204929-s0 \
    runs/reinforce-20261123-204930-s1 runs/reinforce-20261123-204931-s2   # and the three twisty runs
uv run python docs/lessons/scripts/eval_history.py runs/reinforce-20261123-204929-s0 ...
```

[`noisy_vs_mean_laps.py`](scripts/noisy_vs_mean_laps.py) drives each run's final checkpoint two
ways: as the *noisy* policy that training optimized (32 episodes with sampled actions, torch seed
0), and as its *mean* (the deterministic evaluation every lesson's numbers use), and reads the
evaluation history from `metrics.csv`.

| track | seed | run directory | updates | noisy policy: laps 0/1/2 (of 32) | mean policy, final: where it ended, top speed | mean policy: evaluations that lapped |
|---|---|---|---|---|---|---|
| oval | 0 | `runs/reinforce-20261123-204929-s0` | 147 | 9 / 8 / **15** | crash at 558 m, 35.5 m/s | 3 of 30: iterations 25-35 (first at 1,071,712 steps) |
| oval | 1 | `runs/reinforce-20261123-204930-s1` | 140 | 9 / 8 / **15** | crash at 225 m, 28.9 m/s | **0 of 28: never** |
| oval | 2 | `runs/reinforce-20261123-204931-s2` | 129 | 11 / 7 / **14** | crash at 230 m, 29.0 m/s | 7 of 26: iterations 50-80 (first at 2,124,672 steps) |
| twisty | 0 | `runs/reinforce-20261123-204932-s0` | 388 | 32 / 0 / 0 | crash at 277 m, 50.0 m/s | 0 of 78 |
| twisty | 1 | `runs/reinforce-20261123-204933-s1` | 125 | 32 / 0 / 0 | crash at 278 m, 34.0 m/s | 0 of 25 |
| twisty | 2 | `runs/reinforce-20261123-204935-s2` | 149 | 32 / 0 / 0 | crash at 278 m, 43.1 m/s | 0 of 30 |

"Updates" is the number of iterations, one Adam step each. In the noisy column, the episodes with
0 or 1 laps all ended in a crash (or wrong way); the 2-lap ones finished clean.

**On the oval, plain REINFORCE learned to lap, as a noisy driver.** On every seed, the trained
stochastic policy completes both laps in 14-15 of 32 episodes and at least one lap in 21-23. That
is what the algorithm optimizes: the expected return of the *sampled* policy.

**The deterministic mean, which is what every evaluation and the viewer drive, fails on every
seed.** All three final evaluations crash, and seed 1's mean never completed a lap in any of its
28 evaluations. Seeds 0 and 2 did lap for a while (both laps at iterations 25 and 65-80
respectively) and then lost it: seed 0 never lapped again in its remaining 23 evaluations. Why can
the noisy driver lap and its own mean not?

- **Nothing in the algorithm trains the mean as a driver.** The gradient improves the average
  over sampled episodes. The single trajectory you get by always taking the mean is never driven
  in training, so whether it works is an accident. And the noise never shrinks to make the two
  converge: with ~140 Adam steps of `lr = 3e-4`, `log_std` moved from -0.50 to -0.52 (std 0.61 to
  0.59-0.60, `log_std_mean` in `metrics.csv`). An Adam step moves each parameter by roughly
  `lr` (more when its recent gradients agree, so this is a typical size, not a hard bound), and
  [`docs/lessons/scripts/weight_change.py`](scripts/weight_change.py), which rebuilds each run's
  initial actor from its seed, measures the largest change of any of the 5,124 parameters over
  the whole run as 0.045-0.049 (median 0.017-0.020), about `lr × updates`.
- **The noise itself slows the car down, because the brake is twice the engine.** The car turns
  a throttle command `c > 0` into `8c` m/s² and `c < 0` into `16c` m/s² ([car.py#L60](../../racer/car.py#L60),
  `max_accel = 8`, `max_brake = 16` in [`configs/env.toml`](../../configs/env.toml)). So noise
  around a throttle mean does not average out: a sample 0.6 below the mean brakes twice as hard as
  a sample 0.6 above it accelerates. Compare the two drivers where they do the same thing, on the
  opening straight before the first turn (from the `noisy_vs_mean_laps.py` output above):

  | seed | applied throttle before 215 m, noisy / mean | speed at 150 m, noisy (median) / mean |
  |---|---|---|
  | 0 | 0.33 / 0.37 | 22.7 / 29.4 m/s |
  | 1 | 0.32 / 0.38 | 21.5 / 28.8 m/s |
  | 2 | 0.33 / 0.39 | 22.3 / 28.9 m/s |

  The commands differ by only 0.04-0.06 on average, the speeds by 6-7 m/s.
  [`docs/lessons/scripts/noisy_throttle.py`](scripts/noisy_throttle.py) does the arithmetic for a
  throttle mean of 0.35 with std 0.6:

  ```
  uv run python docs/lessons/scripts/noisy_throttle.py 0.35 0.6 150
    samples below 0 (braking): 28.0%, clipped at +1: 13.9%
    mean only, no noise                           E[accel]  2.80 m/s^2  ->  29.0 m/s after 150 m
    noise + clipping, brake as strong as engine   E[accel]  2.48 m/s^2  ->  27.3 m/s after 150 m
    noise + clipping, real brake                  E[accel]  1.67 m/s^2  ->  22.4 m/s after 150 m
  ```

  (`v = sqrt(2 a d)` from standstill, drag ignored.) Clipping at +1 accounts for a small part;
  the asymmetric brake for most of it, and the prediction, 22.4 against 29.0 m/s, matches the
  measured speeds. So the noisy driver arrives at the first turn at about 22 m/s, under the
  24.5 m/s that lesson 02 computed as the most the oval's turns allow at grip 12, and the mean
  arrives at about 29 m/s and runs wide. Training found "this much throttle, plus noise that
  brakes"; driven without the noise, it is too fast. (On seeds 1 and 2 the noisy cars' *top* speed over
  a whole episode, 30.4-30.5 m/s median, is still higher than the mean's 28.9-29.0: the noisy cars
  that survive the first turn go on to drive one or two laps with more straights to accelerate
  on, while the mean's episode ends in that first turn, at its top speed. Seed 0's mean gets
  through the first turn and reaches 35.5 m/s before crashing in the second.)
- **The iterates jump around.** On seed 2, the mean lapped at iterations 50-80 (four of them with
  both laps, down to 28.9 s) and crashed at 85 and every evaluation after; on seed 0 it crashed
  in the second turn (~560 m) or the first (~220 m) for the rest of the run. With gradients as
  noisy as measured above and nothing limiting the step, a single update can move the mean
  driver from one side of "brakes in time" to the other.

**On twisty, plain REINFORCE fails completely.** On all three seeds, 0 of 32 noisy episodes and 0
deterministic evaluations completed a lap. All three mean policies end in the first hairpin at
277-278 m after the 250 m start straight, and so do most noisy episodes (median distance 274-276 m,
quartiles 269-282 m, from the `noisy_vs_mean_laps.py` output above). It is the same trap lesson
04's evolution fell into on seed 0: fast down the straight (seed 0's mean applies full throttle,
1.00 on average before 215 m, and reaches the 50 m/s cap) and no braking in time. Escaping needs a policy that brakes hard at the right
moment *and* turns, and the gradient towards that is buried in noise when every episode ends in
the same wall. REINFORCE's noise, its few updates and its crash-dominated returns are exactly
what the next lesson's critic addresses: A2C laps twisty on all three seeds.

## What to look for in the viewer

```sh
uv run racer watch runs/reinforce-20261123-204931-s2
```

The viewer drives the newest checkpoint's *mean*, clipped, like the evaluations. While a run is in
its crawling phase the live car sits on the start line, its throttle output a hair below zero
(grey in the panel, about -0.02); the progress curve is flat at 0. Later, watch the speed going into the first turn:
the deterministic car of a crashing checkpoint arrives at about 29 m/s and runs wide into the
outer wall. The best-lap curve has points only for the few lapping evaluations, and ghosts appear
only for those. The network panel shows a policy that barely changes: with ~140 updates, the
weights stay close to their initial values.

## Experiments

### 1. Discount: `gamma` 0.95 vs 0.99

**Change.** `γ` sets how far ahead a decision is credited. `γ = 0.99` weighs a reward 100
decisions (6.7 s) ahead at 0.37; `γ = 0.95` weighs one 20 decisions (1.3 s) ahead at 0.36 and one
100 ahead at 0.006. Its effective horizon `1 / (1 - γ)` is 20 decisions instead of 100.

**Predict.** With `γ = 0.95`, the -10 of a crash only reaches back about 1.3 s. Is that enough to
learn to brake before a turn? (Lesson 02: braking from 31 to 24.5 m/s takes 10.6 m, about 0.4 s.)
Sitting still now costs at most `0.01 / 0.05 = 0.2` discounted instead of 1.0, while a crash still
costs 10: does the crawling phase get worse?

**Run.** Oval, **2,000,000 env steps** instead of 4.5M (about 3 min per run, six at a time), seeds
0 and 1, both values, so they are compared at the same budget:

```sh
uv run racer train reinforce --set track=oval --set env_steps_budget=2000000 --set gamma=0.95
uv run racer train reinforce --set track=oval --set env_steps_budget=2000000   # gamma 0.99, the default
uv run python docs/lessons/scripts/noisy_vs_mean_laps.py runs/reinforce-<each run>
```

<details><summary>What we got</summary>

Runs from 2026-11-23, oval, 2M env steps, alongside two other runs (171-177 s each). The `γ = 0.99` runs
repeat the first 2M steps of the main runs: their evaluations match the main runs' (seed 0: both
laps at iteration 25, one lap at 30 and 35).

| `gamma` | seed | run directory | mean policy: evaluations that lapped | noisy policy at 2M: laps 0/1/2, crashes | noisy median top speed |
|---|---|---|---|---|---|
| 0.95 | 0 | `runs/reinforce-20261123-213812-s0` | 1 of 12 (iteration 30, 1 lap) | 31/1/0, 32 crash | 24.4 m/s |
| 0.99 | 0 | `runs/reinforce-20261123-213815-s0` | 3 of 10 (iterations 25-35) | 32/0/0, 19 crash | 12.0 m/s |
| 0.95 | 1 | `runs/reinforce-20261123-213814-s1` | 0 of 11 | 32/0/0, 19 crash | 7.6 m/s |
| 0.99 | 1 | `runs/reinforce-20261123-213816-s1` | 0 of 10 | 32/0/0, 15 crash | 6.1 m/s |

"Crash" counts episodes that ended by crash or wrong way; the rest ran into the time limit.

- **At 2M steps neither value has a noisy policy that laps.** Both seeds are still near the
  crawling phase (seed 1: top speeds of 6-8 m/s); the main runs above lapped as noisy drivers only
  by 4.5M. So the difference between the values is small next to the noise of plain REINFORCE.
- **What does differ, on both seeds: `γ = 0.95` drives faster and crashes more.** Seed 0: median
  top speed 24.4 against 12.0 m/s, 32 of 32 episodes crashing against 19; seed 1: 7.6 against
  6.1 m/s, 19 against 15. That fits the prediction: with a 1.3 s horizon a crash a few seconds
  ahead barely weighs on the decisions that caused it (the acceleration on the straight), while the
  progress those decisions earn right now counts in full.
- **The crawling phase did not get worse; it got shallower.** The prediction pointed the other
  way (sitting still is cheaper relative to a crash), but the 1.3 s horizon also weakens the push
  away from the actions that lead to walls. Seed 1's mean policy never parked with `γ = 0.95`
  (its evaluations drove 60-214 m from iteration 5 on), while with 0.99 it stood at 0.0 m from
  iteration 10 to 25 ([`eval_history.py`](scripts/eval_history.py)). Over iterations 10-28 the
  mean episode length was 1,176 against 1,251 decisions on seed 0 and 1,215 against 1,244 on
  seed 1 ([`metric_window.py`](scripts/metric_window.py) on `episode_len_mean`, every row): with
  0.95 the cars crashed sooner rather than crawled longer.
- Neither got a deterministic mean that laps reliably, as in the main runs.

Two seeds and one short budget: suggestive, not conclusive. The shorter horizon makes the policy
greedier for progress and blinder to crashes, which on this track is the wrong trade.

</details>

### 2. Batch size and learning rate

**Change.** These are the obvious dials if the problem is "too few, too noisy updates":
fewer episodes per batch (more updates, each noisier), a larger learning rate (bigger steps), or
both. All plain REINFORCE: no baseline, normalization, entropy or clipping.

**Predict.** With 8 episodes per batch instead of 32, each update should cost about a quarter as
many decisions, so the run should get about four times as many. Does it learn faster, or does the extra
noise (cosine 0.11 instead of 0.40 above) cancel that? Does a learning rate 10× larger fix the
noisy/mean gap by letting the std shrink?

**Run.** Full budget on the oval, three seeds each (about 7 min per run, six at a time):

```sh
uv run racer train reinforce --set track=oval --set lr=0.001
uv run racer train reinforce --set track=oval --set lr=0.003
uv run racer train reinforce --set track=oval --set episodes_per_batch=8
uv run racer train reinforce --set track=oval --set lr=0.001 --set episodes_per_batch=64
uv run python docs/lessons/scripts/noisy_vs_mean_laps.py runs/reinforce-<each run>
```

These take longer than this lesson's other experiments because they ran as an investigation of
the results above, at the main runs' budget.

<details><summary>What we got</summary>

Runs from 2026-11-23, oval, 4.5M env steps, six at a time. "Noisy" is the final checkpoint's 32
sampled episodes (completed 0/1/2 laps); "mean" the final deterministic evaluation; "lapping
evaluations" counts deterministic evaluations with at least one lap
([`eval_history.py`](scripts/eval_history.py)).

| setting | seed | run directory | updates | noisy laps 0/1/2 | mean, final | lapping evaluations |
|---|---|---|---|---|---|---|
| defaults (`lr` 3e-4, 32 episodes) | 0 | `runs/reinforce-20261123-204929-s0` | 147 | 9/8/15 | crash, 558 m | 3 of 30 |
| | 1 | `runs/reinforce-20261123-204930-s1` | 140 | 9/8/15 | crash, 225 m | 0 of 28 |
| | 2 | `runs/reinforce-20261123-204931-s2` | 129 | 11/7/14 | crash, 230 m | 7 of 26 |
| `lr` 1e-3 | 0 | `runs/reinforce-20261123-211141-s0` | 150 | 1/1/30 | **2 laps**, 25.23 s | 8 of 30 |
| | 1 | `runs/reinforce-20261123-211142-s1` | 149 | 16/9/7 | crash, 559 m | 4 of 30 |
| | 2 | `runs/reinforce-20261123-211143-s2` | 142 | 2/4/26 | **2 laps**, 24.98 s | 12 of 29 |
| `lr` 3e-3 | 0 | `runs/reinforce-20261123-212226-s0` | 164 | 0/0/32 | **2 laps**, 25.05 s | 20 of 33 |
| | 1 | `runs/reinforce-20261123-212229-s1` | 105 | 32/0/0, no crash | parked, 0.0 m | 0 of 21 |
| | 2 | `runs/reinforce-20261123-212231-s2` | 127 | 3/3/26 | **2 laps**, 24.20 s | 11 of 26 |
| 8 episodes per batch | 0 | `runs/reinforce-20261123-211439-s0` | 1,558 | 32/0/0 | crash, 155 m | 5 of 312 |
| | 1 | `runs/reinforce-20261123-211440-s1` | 782 | 30/2/0 | crash, 198 m | 4 of 157 |
| | 2 | `runs/reinforce-20261123-211441-s2` | 702 | 15/12/5 | crash, 210 m | 12 of 141 |
| `lr` 1e-3, 64 episodes | 0 | `runs/reinforce-20261123-212227-s0` | 78 | 4/8/20 | crash, 557 m | 2 of 16 |
| | 1 | `runs/reinforce-20261123-212230-s1` | 77 | 5/6/21 | crash, 558 m | 1 of 16 |
| | 2 | `runs/reinforce-20261123-212232-s2` | 74 | 2/3/26 | crash, 563 m | 2 of 15 |

- **A larger learning rate helps the drivers that move, and can freeze the ones that don't.**
  With `lr` 1e-3 or 3e-3, two of three seeds end with a mean that laps the oval twice (24.2-25.2 s)
  and noisy policies that finish 26-32 of 32 episodes. But no setting laps on every seed. With
  3e-3, seed 1 spent the whole 4.5M steps in the crawling phase: the stronger push away from
  crashes drove its throttle negative (average applied throttle -0.60 at iteration 50, measured with
  `stochastic_vs_mean.py`) and the car stood still in
  every evaluation, 21 of 21. The same push that ends the crawling phase faster on lucky seeds
  makes it deeper on unlucky ones.
- **The update count was not 4× but 5-11×** (702-1,558 updates against 129-147). With fewer
  episodes per batch than the 16 cars, the batch is closed by the first 8 to finish, and once a
  policy crashes early those are short: at seed 0's iteration 300 a batch was only 1,701
  decisions (`batch_bias.py ... 300 8`), against roughly 35,000-43,000 for the defaults. The
  batch rule ties the batch size to episode length: the worse the policy crashes, the smaller
  and more frequent its updates.
- **More, noisier updates made things worse.** With 8 episodes per batch, every seed's mean
  lapped for a while (seed 2 down to 29.9 s, at iteration 135), then stopped lapping for good.
  Seed 0 fell all the way into "flat out, crash in the first turn": its last 293 evaluations
  (iteration 100 to the end) all crash between 145 and 240 m. At its iteration 300,
  `policy_means.py` shows a throttle mean between +1.15 and +2.64 on every state of the
  evaluation episode, so with std 0.6 almost every sample is clipped to full throttle, and all 32
  noisy episodes crash at an applied throttle of 0.99. Once the mean sits far above 1, different
  samples give the same action, and the gradient has almost nothing to compare. Lesson 06 looks
  at this unbounded-mean problem from the other side. Seeds 1 and 2 did not go that far: their
  final means apply 0.71 and 0.49 on the opening straight (`noisy_vs_mean_laps.py`), crash in the
  first turn at 198 and 210 m, and seed 2's noisy policy still drives both laps in 5 of 32
  episodes.
- **More episodes with a larger rate**: no better than the defaults at the end (all three means
  crash at about 560 m, the second turn), with fewer updates (74-78).

Three seeds per setting are few, so read this as what can happen, not as averages. What it does
show: none of these simple dials makes plain REINFORCE reliable; the problems above
(noisy gradients, a driver that is never trained deterministically, iterates that jump) are
structural.

</details>

## Tests worth reading

| test | what it pins |
|---|---|
| [`test_reinforce_gradient_direction`](../../tests/test_reinforce.py#L17-L33) | the loss's gradient moves a Gaussian's mean towards higher reward (a one-step bandit) |
| [`test_every_episode_starts_at_a_spawn`](../../tests/test_reinforce.py#L55-L65) | every batch holds whole episodes, none begun under the previous policy |
| [`test_discounted_returns`](../../tests/test_nets_torch.py#L11-L13) | reward-to-go is the backward recursion `G_t = r_t + γ G_{t+1}` |
| [`test_logprob_matches_distribution`](../../tests/test_nets_torch.py#L16-L22) | the actor's log-probability is the sum of two independent normals' |
| [`test_reinforce_smoke`](../../tests/test_reinforce.py#L36-L47) | a tiny run writes metrics, evaluations and a checkpoint that loads as a policy |

Next: [06 · A2C: a critic, bootstrapping and the truncation fix](06-a2c.md).


