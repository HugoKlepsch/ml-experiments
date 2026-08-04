# ml-experiments

A small bench for training models to play games and comparing them honestly.
Four games, two families of model, a head-to-head arena, a browser viewer, and a
notebook. The games, runner, arena and GA are pure stdlib; the DQN is torch, and
numpy/matplotlib/pandas/jupyter carry the rest, all in a local venv.

The recurring theme, and the reason the arena and the notebook exist at all:
**most of the difficulty is not training a model, it is telling whether one
model is better than another.** Game outcomes are noisy enough that a plausible
gap usually turns out to be luck.

## Setup

```bash
source setup.bash
```

That is the whole thing. It creates `.venv` if missing, installs
`requirements.txt` when it has changed, and activates the environment. Source it
again any time — the common case does no work and returns in a few milliseconds.
From cold it takes about twenty seconds.

It must be **sourced, not executed**, since it changes your current shell. It
refuses to run any other way, and it deliberately avoids `set -e` and `exit` so
that a failure never closes your terminal.

Once active, `python` is the venv's, and `PYTHONPATH` includes the repo root so
the commands below work from any directory. Leave with `deactivate`.

## Quickstart

```bash
source setup.bash

python -m unittest discover -s tests -t .        # 106 tests

python -m train.ga 2048 --name ga                # evolve 2048 weights
python -m train.ga snake --players 2 --name ga   # evolve snake weights
python -m train.dqn snake --players 2 --name dqn # train a snake DQN
python -m train.ga connect4 --name ga            # turn-based, terminal reward
python -m train.dqn kuhn --name dqn              # turn-based, hidden information

# n-step returns, and the raw board through a conv net on the GPU
python -m train.dqn snake --players 2 --n-step 3 --name dqn-n3
python -m train.dqn snake --players 2 --encoding planes --device cuda \
    --batch 256 --buffer 30000 --n-step 3 --envs 64 --name dqn-conv

python -m core.arena snake ga dqn --games 6000   # compare
python serve.py                                  # viewer on :8000
cd notebooks && jupyter lab                      # charts and sweeps
```

Sweeping a parameter, from a notebook or the REPL:

```python
from train.sweep import sweep, round_robin
sweep("dqn", "snake", "hidden", [16, 32, 64, 128, 256], players=2)
sweep("ga",  "snake", "opponent", ["random", "ga", "dqn"], players=2)
round_robin("snake", ["ga", "dqn", "random"], games=2000)
```

## Layout

| path                        | what it is                                                                       |
|-----------------------------|----------------------------------------------------------------------------------|
| `setup.bash`                | sourced environment setup — venv, requirements, `PYTHONPATH`                     |
| `core/game.py`              | the `Game` interface — simultaneous or turn-based, hidden information, features  |
| `core/agent.py`             | `Agent` interface, random/first baselines, and the GA-trained `WeightedAgent`    |
| `core/runner.py`            | plays an episode, optionally recording every frame for replay                    |
| `core/arena.py`             | head-to-head comparison with seat rotation and Wilson intervals                  |
| `core/registry.py`          | look up games and trained models by name                                         |
| `games/g2048.py`            | 2048 (1 player)                                                                  |
| `games/snake.py`            | Snake (N players, simultaneous)                                                  |
| `games/connect4.py`         | Connect Four (2 players, turn-based, terminal-only reward)                       |
| `games/kuhn.py`             | Kuhn poker (2 players, turn-based, hidden information and chance)                |
| `agents/dqn.py`             | torch MLP and conv net + replay buffer, with a JSON model format the UI shares    |
| `train/ga.py`               | genetic algorithm — works on any game                                            |
| `train/dqn.py`              | Double DQN — works on any game                                                   |
| `train/sweep.py`            | vary one parameter, retrain, measure — plus round-robin                          |
| `serve.py` + `static/`      | browser match viewer                                                             |
| `notebooks/analysis.ipynb`  | training curves, distributions, win rates with error bars                        |
| `notebooks/sweeps.ipynb`    | parameter sweeps read against a seed-noise floor                                 |
| `notebooks/optimize.ipynb`  | n-step, flat vs. conv, capacity, vectorised collection, and where the clock goes |
| `models/<game>/<name>.json` | curated models, committed; anything here appears in the UI and arena             |
| `models/<game>/sweeps/`     | sweep output — discoverable the same way, but gitignored                         |

## Results

**2048 — evolved weights vs random**, 400 games on identical seeds:

|          | mean       | median | best   |
|----------|------------|--------|--------|
| `ga`     | **17,741** | 15,998 | 59,628 |
| `random` | 1,084      | 1,036  | 3,060  |

Paired difference +16,657, 95% CI [+15,693, +17,621] — decisive. The agent
reaches the 2048 tile in roughly a quarter of games; random play never passes 256.

**Snake — three-way match**, 900 games, seats rotated:

| agent    | win rate | 95% CI        |
|----------|----------|---------------|
| `ga`     | 50.9%    | 47.6% – 54.2% |
| `dqn`    | 44.9%    | 41.6% – 48.1% |
| `random` | 4.2%     | 3.1% – 5.8%   |

Both trained models crush random. Neither beats the other — and that is the most
instructive result in the repo. At 900 games the GA looked 6.0 points ahead. At
6,000 games:

| agent | win rate | 95% CI        |
|-------|----------|---------------|
| `dqn` | 50.8%    | 49.6% – 52.1% |
| `ga`  | 49.2%    | 47.9% – 50.4% |

The lead did not merely shrink, it **changed hands**: the DQN is now 1.6 points
up, and the 900-game intervals that seemed to favour the GA were wide enough to
contain this all along. **A 6-point win-rate gap over 900 games is not evidence
of anything** — resolving a difference this small takes several thousand games.

That two completely different approaches — seven hand-written features tuned by
evolution, and a neural network learning its own value function from raw
observations — land in a dead heat is a real finding about the difficulty of the
game, not a failure of either method.

**Connect Four — against random**, 2,000 games, seats rotated:

| agent | win rate | 95% CI        |
|-------|----------|---------------|
| `ga`  | 97.9%    | 97.2% – 98.4% |
| `dqn` | 88.9%    | 87.5% – 90.3% |

Head to head the GA wins, and it wins *both* of the two games that exist between
two deterministic agents — see the section on deterministic matchups above for
why that is a much weaker claim than 100% sounds.

The gap is not close, and the reason is the reward. Connect Four scores 0 for
every position until the last move, when it becomes ±1. The GA never has to
solve that: it only needs a fitness number, and `wins_now` and `blocks_win` hand
it most of a tactical policy for free. The DQN has to carry one terminal reward
back across forty plies, and 40,000 episodes gets it to 89% rather than 98%.
**This is the mirror image of the Snake result** — the same two methods, dead
even there, are far apart here, and which one wins is decided by the shape of
the reward rather than by anything intrinsic to either.

The GA also **saturates almost immediately** against a random opponent:

| generation | 1    | 2    | 3    | 4    | … | 25   |
|------------|------|------|------|------|---|------|
| holdout    | 0.82 | 0.94 | 1.00 | 0.99 | … | 1.00 |

Once a candidate beats random every time there is no fitness signal left, and
the remaining 22 generations optimise noise. Anything better has to be selected
against a stronger opponent (`--opponent ga`), which is what `--opponent` is for.

**Kuhn poker**, 40,000 hands, seats rotated. Chips per hand, which is the unit
that means something here:

| matchup       | chips/hand | 95% CI             | win rate |
|---------------|------------|--------------------|----------|
| `ga` v random | +0.4708    | +0.4556 … +0.4860  | 69.1%    |
| `dqn` v random| +0.3459    | +0.3316 … +0.3602  | 60.8%    |
| `ga` v `dqn`  | +0.0895    | +0.0733 … +0.1057  | 58.5%    |

The GA beats the DQN by about a tenth of a chip a hand, and the interval clears
zero comfortably. Two caveats that make this game worth having:

**There is an exact answer to check against.** Kuhn is small enough to solve on
paper: against perfect play the first player loses **1/18 ≈ 0.056 chips a hand**.
Every other result in this repo is one model measured against another model of
unknown strength; this is the only one with a known optimum behind it.

**And 1/18 is brutally expensive to measure.** A hand pays out ±1 or ±2, so the
per-hand standard deviation is about 1.6 — roughly thirty times the edge being
measured. Resolving a 1/18 difference at 95% takes **about 3,000 hands**, and
that is for a gap the size of the entire game value. It is the cheapest available
demonstration of the thing this whole bench is about.

Neither agent can actually reach the optimum, and that is a property of the
agents rather than of the training. Kuhn's optimal strategy is *mixed* — it
bluffs the jack at a specific frequency — and a `WeightedAgent` with fixed
weights is deterministic, so no weight vector reaches it. `temperature` on that
agent is what buys the randomisation back.

## Parameter sweeps

`notebooks/sweeps.ipynb` retrains a model while varying one parameter and measures
the result. Its organising idea is that **a sweep is unreadable without a noise
floor**: two runs differing only in random seed also produce different numbers, so
you need to know that spread before believing any of the others.

Sweeping the DQN's hidden width on Snake (2,500 episodes, 300 games per measurement):

| hidden | mean score | verdict against the noise floor        |
|--------|------------|----------------------------------------|
| 16     | 2.61       | **below — plausibly a real loss**      |
| 32     | 4.62       | inside — indistinguishable from a seed |
| 64     | 12.84      | inside — indistinguishable from a seed |
| 128    | 22.28      | **above — plausibly a real gain**      |
| 256    | 22.58      | **above — plausibly a real gain**      |

That looks like a clean monotonic curve. But five runs at a *fixed* `hidden=64`,
differing only by seed, scored 4.42, 9.23, 12.34, 12.84 and 18.52 — a spread of
14.10 against a sweep spread of 19.97. Three of the five settings separate from
that band — 16 below it, 128 and 256 above — and 32 and 64 sit inside it, so the
apparent step between those two is seed luck. Note what that costs: a five-point
sweep resolved into three groups, not five.

Doing this properly means k seeds per setting at k times the cost, which is exactly
why so many published sweeps report one run per cell.

The GA opponent sweep — does training against a harder opponent help? — came back
negative: scored against a common `random` opponent, training against `random`,
`ga` and `dqn` gave 23.66, 23.90 and 21.58, with fully overlapping intervals.

`notebooks/optimize.ipynb` does the same for n-step and the conv encoding, and
both came back negative on Snake at a 1,500-episode budget. Reported here for the
same reason as the ramp above:

**n-step** (1,500 episodes, 300 games per measurement, seed band 11.60–18.03):

| n_step | mean score | verdict                                   |
|--------|------------|-------------------------------------------|
| 1      | 18.03      | inside the band                           |
| 2      | 18.46      | above it by 0.43, on a band 6.43 wide     |
| 3      | 16.93      | inside                                    |
| 5      | 14.03      | inside, near the bottom                   |
| 8      | 10.10      | **below — a real loss**                   |

n=2 clears the band by less than a fifteenth of its width, which is not a result.
The loss at n=8 is. That is the predicted shape with only the downside visible,
and the reason is the game: **Snake's reward is dense.** Food, a per-tick survival
bonus and a death penalty arrive continuously, so there is almost no delayed
credit for n-step to rescue, while the off-policy bias is charged in full. Connect
Four — one reward, forty plies from the opening — is where this should pay, and is
the obvious next thing to sweep.

**The conv**, same budget, everything but the encoding held constant:

| encoding | params  | train  | mean score |
|----------|---------|--------|------------|
| `flat`   | 19,588  | 13.5s  | **21.49**  |
| `planes` | 673,508 | 76.3s  | 4.01       |

The intervals do not overlap: at this budget the conv is decisively *worse*, and
widening it from 16 to 128 channels does nothing (4.75, 3.91, 4.79, 5.05 — all
inside each other's error bars, at 3x the training time).

This is the most useful negative result in the repo, because it is the one people
get wrong. The flat encoding is not raw data — it is nineteen numbers that
already encode danger, free space and food direction, distilled by someone who
knows the game. The conv has to *rediscover* all of that from the board, and a
few thousand episodes is nowhere near enough. **Capacity is a loan against future
data, and it is expensive until the data arrives.** The comparison worth trusting
is at the budget where both have converged, which is what `FAST = False` is for.

The notebook also splits a training run's wall clock into collecting experience
and updating the network, which is the measurement that says what to optimise
next. 150 episodes:

| configuration  | total  | collecting | per update |
|----------------|--------|------------|------------|
| `flat` / cpu   | 0.9s   | 26%        | 0.67 ms    |
| `planes` / cpu | 233.9s | 0%         | 224.04 ms  |
| `planes` / cuda| 8.3s   | 9%         | 8.28 ms    |

The GPU turns a 4-minute conv run into an 8-second one. Note where that leaves
the split, though: collection is **9%** of the clock, so making it free could
only ever return another 9%. See the next section, which measures exactly that
and finds what the arithmetic predicts.

## The two model families

**`WeightedAgent` + GA.** You write the features; evolution prices them. Needs
only a fitness score, so it never has to solve credit assignment. Cheap, easy to
interpret, and limited by the quality of the features you thought of.

**`DQNAgent`.** Learns a value for each action from the raw observation vector,
via Double DQN with a replay buffer and a target network. No feature engineering,
but far more hyperparameters and much harder to debug. The network is an
`nn.Module` wrapping an `nn.Sequential`, so torch owns the layers, the autograd
and the optimiser, and `train/dqn.py` is left holding only the DQN logic —
targets, action masking, bootstrapping — which is the part worth reading.

Models are JSON, not `torch.save` output: the `state_dict` goes to disk as
nested lists next to the config and the training history the notebooks plot.
One readable, diffable format for every model in `models/`, GA and DQN alike,
and `core/registry.py` does not need to know which kind a file is until it opens
it. Each net also records an `arch` dict that is exactly its constructor
arguments, so the loader rebuilds an MLP or a conv net without being told which
to expect. Models written before `arch` existed still load.

## Observations, architectures and n-step

Three things worth separating, because calling them all "the model" hides the
choices that matter.

**The learning algorithm** — GA or DQN — is how weights get updated. **The
architecture** is what the network is. **The observation encoding** is what it
gets to see. All three vary independently.

### n-step returns

`--n-step` (1 for textbook DQN). At n=1 a transition carries one real reward and
then starts guessing; at n=3 it carries three. The point is how fast reward
information travels backwards. A snake that traps itself and dies ten ticks later
learns, at n=1, only that the final state was fatal — the move that actually did
it waits ten more rounds of bootstrapping to hear anything.

The honest caveat: an n-step return is **biased off-policy**, because the
intermediate actions came from an older, more exploratory policy and nothing
corrects for it. The bias grows with n, so the curve improves and then turns
back down. Where it turns is a property of the game, which is why
`notebooks/optimize.ipynb` measures it rather than copying 3 from a paper.

The window is counted in the **learner's decisions**, not ticks — the same
choice `collect_episode` already made for one-step transitions, and what makes
n-step mean the same thing in Connect Four as in Snake.

### Observation encodings

A game may offer several descriptions of the same position. `Game.encodings`
lists them, `observe(state, player, encoding)` takes one, `obs_spec(encoding)`
reports the resulting length and grid shape. Only Snake currently offers more
than one:

| encoding | what the network sees                                                        |
|----------|------------------------------------------------------------------------------|
| `flat`   | 19 hand-crafted floats — danger per direction, flood-filled free space, food vector |
| `planes` | 8 binary 12x12 board planes, **plus the 19 flat features appended**          |

Two design rules keep this from leaking everywhere. `observe` always returns a
**flat list** whatever the encoding — a grid lays its planes out first in
C-order and any scalars follow — so the replay buffer stays a plain 2-D array
and only the network reshapes. And the **agent** remembers its encoding, not the
game, so the arena builds one `SnakeGame` and a flat MLP and a conv net can sit
down at it together.

The planes are own head/body/tail, enemy head/body/tail, food, and an all-ones
mask over the board. That last one exists because convolutions pad with zeros:
without it a cell past the edge is indistinguishable from an empty one, and with
it the padding itself is what marks the wall.

`planes` **appends** the flat vector rather than replacing it. `free_space` is a
flood fill over the whole board, and a three-layer tower of 3x3 filters sees a
7-cell window — it cannot compute that however long it trains. Feeding both
usually beats either alone.

### The conv net

`ConvNet` is a tower of 3x3 convolutions over the grid, a 1x1 convolution to
squeeze channels, then a linear head that the scalar side-channel is
concatenated into. The argument for it over a wider MLP on the same input is
**translation equivariance**: a convolution learns "food one cell to my left"
once, rather than once per board square.

Three deliberate omissions. **No pooling** — pooling discards position, and in
Snake position is the problem. **No normalisation layers** — BatchNorm sees
different batch statistics in the online and target networks and makes them
disagree for reasons unrelated to learning. And the **1x1 reduce** before
flattening, because flattening 64 channels of a 12x12 board straight into a
128-wide layer is 1.2M parameters, most of the net, for nothing.

### On the GPU

`--device cuda`. Worth knowing what it does and does not buy. One forward pass
over a batch of 256, on an RTX 2080:

| network | cpu     | cuda    | per sample, cuda |
|---------|---------|---------|------------------|
| MLP     | 0.09 ms | 0.03 ms | 0.12 µs          |
| conv    | 58.8 ms | 1.29 ms | 5.1 µs           |

For scale, one game step plus one observation is about 60 µs, all of it Python.

The MLP is free either way — its per-sample cost is a couple of orders of
magnitude under the environment's, so moving it to the GPU changes nothing you
can measure, and at batch 1 the GPU is often *slower* because a kernel launch
costs more than the arithmetic it carries. That is why `train/dqn.py` still
defaults to `cpu`. The conv is the opposite, and its 45x only appears at a real
batch size.

The general form: **a GPU is a throughput device, so it pays only once the work
is batchy.** The conv defaults pair `--device cuda` with `--batch 256`.

A grid observation is ~60x wider than a flat one, so the replay buffer stores
observations as float16 for grid encodings and float32 otherwise — two float32
copies at the default 100k capacity would be 940 MB of RAM doing nothing useful.
Observations are inputs, not parameters; half precision costs about three
decimal digits of something already normalised to roughly [0, 1].

### Vectorised collection

`--envs N` plays N episodes concurrently. Nothing about the game gets faster —
stepping N Python games costs exactly N times stepping one. What changes is that
every agent is asked for all its pending decisions at once, so the network does
one forward pass over N positions instead of N passes over one. At batch 1 a GPU
spends longer launching the kernel than running it, so this is the difference
between using the device and merely owning it.

`collect_batch` groups pending decisions **by agent object, not by seat**, which
is what makes it work under seat rotation: the learner sits in different seats in
different episodes but is one object, so its decisions still form one batch. The
collector also builds each observation once and hands it to the agent, rather
than letting `act` rebuild the vector the replay buffer is about to store — worth
as much here as the batching, since a planes observation costs about as much as a
forward pass.

`--envs` is a **pure throughput knob**: same episodes, same gradient steps, same
replay ratio, same seat rotation. `tests/test_vectorised.py` holds it to that,
differentially — batched collection must store exactly the transitions serial
collection would, across all four games, every n-step setting, both net types and
mixed seat assignments. If that property broke, `--envs` would be silently
changing what a run learns from and every result in the repo would become
conditional on a setting nobody thinks of as an experiment.

**Collection throughput**, 256 Snake episodes, transitions per second:

| envs | flat MLP / cpu | conv / cpu    | conv / cuda   |
|------|----------------|---------------|---------------|
| 1    | 7,799          | 2,092         | 2,821         |
| 4    | 9,395          | 2,752         | 4,694         |
| 16   | 10,990         | 3,281         | 6,356         |
| 64   | 12,093 (1.55x) | 3,442 (1.65x) | 6,661         |
| 256  | 11,942         | 3,417         | 7,531 (2.67x) |

It does what it claims — 2.7x on the GPU conv, and it keeps climbing where both
CPU variants peak early and then flatten or *decline*, since a CPU convolution at
batch 256 is no faster per sample than at batch 1 and the extra bookkeeping is
pure loss. That is section 1's measurement showing up from the other side.

**But it barely moves total training time**, and that is the result worth
recording. End to end on `planes`/`cuda`, 200 episodes, replay ratio held fixed:

| updates/episode | envs=1 | envs=64 | speedup |
|-----------------|--------|---------|---------|
| 8 (default)     | 7.14s  | 7.44s   | 0.96x   |
| 4               | 4.18s  | 3.94s   | 1.06x   |
| 2               | 2.51s  | 2.14s   | 1.17x   |
| 1               | 1.57s  | 1.23s   | 1.27x   |

Collection was **9%** of that configuration's clock, so Amdahl caps the win near
1.1x at the default and it comes in at 0.96x — the batching overhead is real and
at a high replay ratio there is nothing for it to buy. **Training here is
update-bound**, and a faster collector cannot fix that. The gain only appears
once you stop buying eight gradient steps per episode of fresh experience.

Two things it earns anyway, which the wall-clock table does not show. It makes a
**lower replay ratio affordable** — and fewer gradient steps per fresh transition
is usually better for final performance, not merely faster, so the bottom row of
that table is the interesting one rather than the top. And it moves the
bottleneck onto **pure-Python game stepping**: flat/cpu plateaus near 12,000
transitions per second, against roughly 16,700 for one `step` plus one `observe`
and nothing else. Getting past *that* needs Snake's dynamics reimplemented over
batched arrays — a much larger change than this one, and worth it only if
collection ever becomes the thing to optimise. On this evidence it is not.

## Practices baked in

- **Common random numbers.** Every agent faces the same seeds. For one-player
  games this makes the comparison paired, which shrinks the error bar a lot.
- **Seat rotation.** Agents cycle through every seat so position never masquerades
  as skill.
- **Wilson intervals** on every win rate. A bare percentage invites over-reading.
- **A holdout set** the GA never selects on, so the training curve is comparable
  across generations.
- **A games-per-eval ramp** (12 → 60): cheap while differences are large, precise
  once the population converges. The earlier flat-budget run is kept as
  `models/2048/ga-flat.json` for comparison.

  Worth noting that on 2048 the ramp **did not measurably help**: paired over 600
  games, `ga` beats `ga-flat` by 377 points, 95% CI [-695, +1449] — not
  significant. The theory is sound, but the binding constraint here is the
  feature set, not selection precision. Reported rather than quietly dropped,
  because a bench that only records its wins is not much of a bench.

## The game interface

Every game is a `core.game.Game`. Add it to `_games()` in `core/registry.py` and
both trainers, the arena, the viewer and every chart work on it unchanged.

**Simultaneous games** — 2048, Snake — implement `reset`, `legal_actions`,
`step`, `is_terminal`, `scores`, `observe`, `render` and `action_features`.
`step` takes one action per player and advances a tick.

**Turn-based games** — Connect Four, Kuhn poker — subclass `TurnBasedGame` and
implement `current_player`, `moves` and `play`, each phrased for the single
player to move. The action-tuple plumbing is handled for you: seats that are not
to move pass `NO_ACTION`, so from the outside a turn-based game is just one where
all but one seat has no legal action. That is why nothing downstream needed a
special case.

**Hidden information** is expressed by `view(state, player)`, which returns what
that player is allowed to know. The runner hands agents the *view*, never the
state, so a poker agent cannot physically read its opponent's card. The contract
that makes this work: every method an agent can reach — `legal_actions`,
`action_features`, `observe` — must accept a view as well as a full state, and
give the same answer on both. Perfect-information games return the state
unchanged and pay nothing. `tests/test_turn_based.py` checks all of it.

Three rules that will bite otherwise: states must be immutable, since the runner
keeps old ones for replay; `action_features` / `observe` must return exactly as
many values as `feature_names` / `obs_size` declare; and `render` is the
*spectator* view, built only for finished replays, so it may reveal what `view`
hides.

### What turn-based games broke

Worth recording, because none of it was visible until a real turn-based game
existed:

- **The DQN collector was storing one transition per tick.** In a turn-based
  game the opponent moves in between, so the position a learner bootstraps from
  is not the next tick — it is its own next decision. Worse, in Connect Four the
  reward for losing arrives on the *opponent's* ply, so a per-tick collector
  attributed it to nobody and the learner never saw a defeat at all.
  `collect_episode` now spans decision to decision, accumulating reward across
  the plies in between. Simultaneous games are unaffected, because there the two
  are the same thing — there is a test pinning that.
- **The arena's confidence intervals were lying.** See below.

## Deterministic games break the error bars

Connect Four has no chance in it. Two saved models are deterministic. So every
seed produces *the same game*, and 4,000 games are two games recorded 2,000
times each:

```
connect4: 200 games, seats rotated
  ga                 win 100.0%  [98.1%, 100.0%]   mean score       1.0
  dqn                win   0.0%  [ 0.0%,  1.9%]   mean score      -1.0

  WARNING: every seed produced the same game. Neither the game nor the agents
  have any randomness in them, so this is 2 distinct match(es) repeated, not
  200 samples. Ignore the interval above [...]
```

That `[98.1%, 100.0%]` is computed from a sample size of two. The arena now
detects the case — every seating yielding a single outcome across all its seeds —
and says so rather than printing a confident number. It stays quiet when either
the game deals cards or an agent randomises, which is the whole point: this is
the one situation where common random numbers buy you nothing at all.

## Design notes

- 2048 boards are flat 16-tuples of exponents, so all 65,536 row transitions are
  precomputed. ~45 ms/game with a heuristic agent.
- Snake resolves collisions after every snake has moved: head-on kills both
  regardless of length, and entering a cell a tail is vacating is survivable
  unless that snake just ate. Snakes starve after 100 foodless ticks so two
  cautious agents cannot circle forever.
- Connect Four precomputes all 69 four-in-a-row windows, so threat counting is a
  flat sweep rather than nested bounds checks. Scores are zero-sum and
  terminal-only: +1/-1, and 0 each for a draw, which `winners` reports as a tie
  and the arena splits half a win each way.
- Kuhn poker is the two-action formulation: `pass` is a check when nothing is
  owed and a fold when facing a bet, `bet` is a bet or a call. `view` blanks the
  opponent's card, and `scores` is only ever called on the true state — a view
  is missing a card and could not settle a showdown.
- The viewer is stdlib `http.server` bound to localhost, and reads `models/` on
  every request — train something new and it shows up on refresh.
- Illegal actions are masked, never penalised: a model must not be able to pick
  one no matter what it predicts.
