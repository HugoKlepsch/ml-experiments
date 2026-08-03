# ml-experiments

A small bench for training models to play games and comparing them honestly.
Two games, two families of model, a head-to-head arena, a browser viewer, and a
notebook. Pure stdlib except numpy/matplotlib/pandas/jupyter in a local venv.

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

python -m unittest discover -s tests -t .        # 55 tests

python -m train.ga 2048 --name ga                # evolve 2048 weights
python -m train.ga snake --players 2 --name ga   # evolve snake weights
python -m train.dqn snake --players 2 --name dqn # train a snake DQN

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
| `core/game.py`              | the `Game` interface — N simultaneous players, per-action features, observations |
| `core/agent.py`             | `Agent` interface, random/first baselines, and the GA-trained `WeightedAgent`    |
| `core/runner.py`            | plays an episode, optionally recording every frame for replay                    |
| `core/arena.py`             | head-to-head comparison with seat rotation and Wilson intervals                  |
| `core/registry.py`          | look up games and trained models by name                                         |
| `games/g2048.py`            | 2048 (1 player)                                                                  |
| `games/snake.py`            | Snake (N players, simultaneous)                                                  |
| `agents/dqn.py`             | MLP + replay buffer, backward pass written out in numpy                          |
| `train/ga.py`               | genetic algorithm — works on any game                                            |
| `train/dqn.py`              | Double DQN — works on any game                                                   |
| `train/sweep.py`            | vary one parameter, retrain, measure — plus round-robin                          |
| `serve.py` + `static/`      | browser match viewer                                                             |
| `notebooks/analysis.ipynb`  | training curves, distributions, win rates with error bars                        |
| `notebooks/sweeps.ipynb`    | parameter sweeps read against a seed-noise floor                                 |
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
| `dqn`    | 48.5%    | 45.3% – 51.8% |
| `ga`     | 47.0%    | 43.7% – 50.2% |
| `random` | 4.5%     | 3.3% – 6.1%   |

Both trained models crush random. Neither beats the other — and that is the most
instructive result in the repo. At 900 games the DQN looked 3.6 points ahead. At
6,000 games:

| agent | win rate | 95% CI        |
|-------|----------|---------------|
| `dqn` | 50.5%    | 49.2% – 51.8% |
| `ga`  | 49.5%    | 48.2% – 50.8% |

The gap collapsed to 1.0 point and both intervals now straddle 50%. The earlier
lead was noise. **A 3.6-point win-rate gap over 900 games is not evidence of
anything** — resolving a difference that small takes several thousand games.

That two completely different approaches — seven hand-written features tuned by
evolution, and a neural network learning its own value function from raw
observations — land in a dead heat is a real finding about the difficulty of the
game, not a failure of either method.

## Parameter sweeps

`notebooks/sweeps.ipynb` retrains a model while varying one parameter and measures
the result. Its organising idea is that **a sweep is unreadable without a noise
floor**: two runs differing only in random seed also produce different numbers, so
you need to know that spread before believing any of the others.

Sweeping the DQN's hidden width on Snake (2,500 episodes, 300 games per measurement):

| hidden | mean score | verdict against the noise floor        |
|--------|------------|----------------------------------------|
| 16     | 4.16       | inside — indistinguishable from a seed |
| 32     | 5.54       | inside — indistinguishable from a seed |
| 64     | 10.11      | inside — indistinguishable from a seed |
| 128    | 22.05      | **above — plausibly a real gain**      |
| 256    | 22.52      | **above — plausibly a real gain**      |

That looks like a clean monotonic curve. But five runs at a *fixed* `hidden=64`,
differing only by seed, scored 3.79, 10.11, 11.04, 15.03 and 15.54 — a spread of
11.75 against a sweep spread of 18.36. Only the top two settings clear it. The
apparent difference between 16, 32 and 64 is seed luck.

Doing this properly means k seeds per setting at k times the cost, which is exactly
why so many published sweeps report one run per cell.

The GA opponent sweep — does training against a harder opponent help? — came back
negative: scored against a common `random` opponent, training against `random`,
`ga` and `dqn` gave 23.66, 23.90 and 23.22, with fully overlapping intervals.

## The two model families

**`WeightedAgent` + GA.** You write the features; evolution prices them. Needs
only a fitness score, so it never has to solve credit assignment. Cheap, easy to
interpret, and limited by the quality of the features you thought of.

**`DQNAgent`.** Learns a value for each action from the raw observation vector,
via Double DQN with a replay buffer and a target network. No feature engineering,
but far more hyperparameters and much harder to debug. `tests/test_core.py`
checks the hand-written backward pass against finite differences, which is the
first thing to suspect when a from-scratch net will not learn.

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

## Adding a game

Implement `core.game.Game` and add it to `_games()` in `core/registry.py`. You
need `reset`, `legal_actions`, `step`, `is_terminal`, `scores`, `observe`,
`render`, and `action_features`. Both trainers, the arena, the viewer and every
chart then work on it unchanged.

Two rules that will bite otherwise: states must be immutable, since the runner
keeps old ones for replay; and `action_features` / `observe` must return exactly
as many values as `feature_names` / `obs_size` declare (there is a test for this).

## Design notes

- 2048 boards are flat 16-tuples of exponents, so all 65,536 row transitions are
  precomputed. ~45 ms/game with a heuristic agent.
- Snake resolves collisions after every snake has moved: head-on kills both
  regardless of length, and entering a cell a tail is vacating is survivable
  unless that snake just ate. Snakes starve after 100 foodless ticks so two
  cautious agents cannot circle forever.
- The viewer is stdlib `http.server` bound to localhost, and reads `models/` on
  every request — train something new and it shows up on refresh.
- Illegal actions are masked, never penalised: a model must not be able to pick
  one no matter what it predicts.
