"""N-player simultaneous Snake.

All snakes move on the same tick, which is what makes this a genuine
multi-agent game rather than N independent ones: the board your opponent
occupies next tick depends on a decision they are making right now, not one you
can observe first.

Collision rules, resolved after every snake has moved:
  * a head off the board dies;
  * two heads entering the same cell kill each other, regardless of length;
  * a head entering any surviving body segment dies;
  * a head entering the cell a tail is vacating survives, unless that snake
    just ate and therefore did not move its tail.

Snakes also starve: going `starve_limit` ticks without food is fatal, which
stops two cautious agents from circling forever.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from core.game import Game

UP, DOWN, LEFT, RIGHT = range(4)
_DELTA = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}
_OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

FOOD_VALUE = 10.0     # score per food
TICK_VALUE = 0.01     # score per tick survived, breaks ties toward longevity
DEATH_PENALTY = -5.0  # applied once, to the RL reward only


@dataclass(frozen=True)
class SnakeState:
    bodies: tuple           # per player: tuple of (x, y), head first
    alive: tuple            # per player: bool
    directions: tuple       # per player: last direction moved
    food: frozenset
    eaten: tuple            # per player: food count
    ticks: tuple            # per player: ticks survived
    hunger: tuple           # per player: ticks since last food
    step: int


class SnakeGame(Game):
    name = "snake"
    action_names = ("up", "down", "left", "right")

    # `flat` is the hand-crafted vector; `planes` is the raw board for a conv
    # net, with the flat vector appended to it. See `_planes`.
    encodings = ("flat", "planes")
    PLANE_NAMES = (
        "my_head", "my_body", "my_tail",
        "enemy_heads", "enemy_bodies", "enemy_tails",
        "food", "on_board",
    )

    feature_names = (
        "dies",          # this move is immediately fatal
        "eats",          # lands on food this tick
        "food_closer",   # reduction in distance to nearest food
        "free_space",    # reachable area from the new head, flood filled
        "enemy_near",    # proximity of the nearest rival head
        "length_edge",   # our length versus the longest rival
        "center",        # closeness to the middle of the board
    )

    def __init__(self, width=12, height=12, num_players=2, starve_limit=100):
        self.width = width
        self.height = height
        self.num_players = num_players
        self.starve_limit = starve_limit
        self.obs_size = 4 + 4 + 4 + 2 + 3 + 2
        # The grid, then the whole flat vector after it as a scalar channel.
        self.planes_size = (
            len(self.PLANE_NAMES) * self.height * self.width + self.obs_size
        )

    def obs_spec(self, encoding=None):
        self._check_encoding(encoding)
        if encoding == "planes":
            return self.planes_size, (len(self.PLANE_NAMES), self.height, self.width)
        return self.obs_size, None

    # --- setup ------------------------------------------------------------

    def _start_slots(self):
        w, h = self.width, self.height
        return [
            (w // 4, h // 4), (3 * w // 4, 3 * h // 4),
            (3 * w // 4, h // 4), (w // 4, 3 * h // 4),
            (w // 2, h // 4), (w // 2, 3 * h // 4),
            (w // 4, h // 2), (3 * w // 4, h // 2),
        ]

    def reset(self, rng):
        slots = self._start_slots()
        if self.num_players > len(slots):
            raise ValueError(f"at most {len(slots)} players supported")

        bodies, directions = [], []
        for i in range(self.num_players):
            hx, hy = slots[i]
            # Face the middle so nobody opens the game pointed at a wall.
            direction = RIGHT if hx < self.width / 2 else LEFT
            dx, dy = _DELTA[direction]
            bodies.append(tuple((hx - dx * k, hy - dy * k) for k in range(3)))
            directions.append(direction)

        occupied = {cell for body in bodies for cell in body}
        food = set()
        while len(food) < self.num_players:
            food.add(self._free_cell(occupied | food, rng))

        return SnakeState(
            bodies=tuple(bodies),
            alive=(True,) * self.num_players,
            directions=tuple(directions),
            food=frozenset(food),
            eaten=(0,) * self.num_players,
            ticks=(0,) * self.num_players,
            hunger=(0,) * self.num_players,
            step=0,
        )

    def _free_cell(self, occupied, rng):
        free = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) not in occupied
        ]
        if not free:
            raise ValueError("board is full")
        return rng.choice(free)

    def _in_bounds(self, cell):
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    # --- dynamics ---------------------------------------------------------

    def legal_actions(self, state, player):
        """Every direction except a straight reversal into your own neck."""
        if not state.alive[player]:
            return []
        body = state.bodies[player]
        blocked = _OPPOSITE[state.directions[player]] if len(body) > 1 else None
        return [a for a in range(4) if a != blocked]

    def step(self, state, actions, rng):
        bodies = list(state.bodies)
        alive = list(state.alive)
        directions = list(state.directions)
        eaten = list(state.eaten)
        ticks = list(state.ticks)
        hunger = list(state.hunger)
        food = set(state.food)

        moving = [i for i in range(self.num_players) if alive[i]]
        new_heads, grew = {}, {}
        for i in moving:
            action = actions[i]
            if action not in self.legal_actions(state, i):
                action = state.directions[i]
            dx, dy = _DELTA[action]
            hx, hy = bodies[i][0]
            new_heads[i] = (hx + dx, hy + dy)
            grew[i] = new_heads[i] in food
            directions[i] = action

        # Provisional post-move bodies, before working out who died.
        moved = {}
        for i in moving:
            body = (new_heads[i],) + bodies[i]
            moved[i] = body if grew[i] else body[:-1]

        dead = set()
        for i in moving:
            if not self._in_bounds(new_heads[i]):
                dead.add(i)
        for i in moving:
            for j in moving:
                if i < j and new_heads[i] == new_heads[j]:
                    dead.add(i)
                    dead.add(j)
        for i in moving:
            if i in dead:
                continue
            for j in moving:
                # Compare against bodies excluding heads; heads are handled above.
                if new_heads[i] in moved[j][1:]:
                    dead.add(i)
                    break

        for i in moving:
            hunger[i] = 0 if grew[i] else hunger[i] + 1
            if hunger[i] >= self.starve_limit:
                dead.add(i)

        for i in moving:
            if i in dead:
                alive[i] = False
                continue
            bodies[i] = moved[i]
            ticks[i] += 1
            if grew[i]:
                eaten[i] += 1
                food.discard(new_heads[i])

        occupied = {
            cell for i in range(self.num_players) if alive[i] for cell in bodies[i]
        }
        target = sum(1 for a in alive if a) or 1
        while len(food) < target and len(occupied) + len(food) < self.width * self.height:
            food.add(self._free_cell(occupied | food, rng))

        return SnakeState(
            bodies=tuple(bodies),
            alive=tuple(alive),
            directions=tuple(directions),
            food=frozenset(food),
            eaten=tuple(eaten),
            ticks=tuple(ticks),
            hunger=tuple(hunger),
            step=state.step + 1,
        )

    def is_terminal(self, state):
        living = sum(1 for a in state.alive if a)
        # With rivals present the game is over once a single snake remains.
        return living == 0 or (self.num_players > 1 and living <= 1)

    def alive(self, state, player):
        return state.alive[player]

    def scores(self, state):
        return tuple(
            FOOD_VALUE * state.eaten[i] + TICK_VALUE * state.ticks[i]
            for i in range(self.num_players)
        )

    def reward(self, state, next_state, player):
        delta = self.scores(next_state)[player] - self.scores(state)[player]
        if state.alive[player] and not next_state.alive[player]:
            delta += DEATH_PENALTY
        return delta

    # --- features ---------------------------------------------------------

    def _obstacles(self, state, skip_tail_of=None):
        """Cells that would kill a head entering them this tick."""
        cells = set()
        for i in range(self.num_players):
            if not state.alive[i]:
                continue
            body = state.bodies[i]
            # The tail moves out of the way unless that snake is about to grow.
            cells.update(body if i == skip_tail_of else body[:-1])
        return cells

    def _flood(self, start, blocked, limit=60):
        """Reachable free cells from `start`, capped so this stays cheap."""
        seen = {start}
        queue = deque([start])
        while queue and len(seen) < limit:
            x, y = queue.popleft()
            for dx, dy in _DELTA.values():
                cell = (x + dx, y + dy)
                if cell in seen or cell in blocked or not self._in_bounds(cell):
                    continue
                seen.add(cell)
                queue.append(cell)
        return len(seen)

    def _nearest_food(self, state, cell):
        if not state.food:
            return self.width + self.height
        return min(abs(cell[0] - f[0]) + abs(cell[1] - f[1]) for f in state.food)

    def action_features(self, state, player, action):
        body = state.bodies[player]
        dx, dy = _DELTA[action]
        head = (body[0][0] + dx, body[0][1] + dy)
        span = self.width + self.height

        blocked = self._obstacles(state, skip_tail_of=player)
        fatal = not self._in_bounds(head) or head in blocked
        if fatal:
            # Nothing past the wall is meaningful; report the death and stop.
            return (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        eats = head in state.food
        closer = self._nearest_food(state, body[0]) - self._nearest_food(state, head)

        rivals = [
            state.bodies[i][0]
            for i in range(self.num_players)
            if i != player and state.alive[i]
        ]
        enemy = min(
            (abs(head[0] - r[0]) + abs(head[1] - r[1]) for r in rivals), default=span
        )
        longest = max(
            (len(state.bodies[i]) for i in range(self.num_players)
             if i != player and state.alive[i]),
            default=0,
        )
        centre = abs(head[0] - self.width / 2) + abs(head[1] - self.height / 2)

        return (
            0.0,
            1.0 if eats else 0.0,
            closer / 2.0,
            self._flood(head, blocked) / 60.0,
            1.0 / (1.0 + enemy),
            (len(body) - longest) / 10.0,
            1.0 - centre / span,
        )

    def observe(self, state, player, encoding=None):
        self._check_encoding(encoding)
        flat = self._flat(state, player)
        if encoding == "planes":
            return self._planes(state, player) + flat
        return flat

    def _planes(self, state, player):
        """The board itself, as binary planes, flattened in C-order.

        A convolution is *translation-equivariant*: it learns "food one cell to
        my left" once, rather than once per square. That is the whole reason to
        hand a net the grid instead of the summary in `_flat` — and the reason
        the summary is appended anyway, since `free_space` is a flood fill over
        the whole board and no small stack of 3x3 filters can compute it.

        Planes are ordered so that everything about the player to move comes
        first, which makes them readable when printed and lets a net trained on
        a 2-player board be reused on more seats without reindexing.
        """
        w, h = self.width, self.height
        cells = w * h
        grid = [0.0] * (len(self.PLANE_NAMES) * cells)

        def mark(plane, cell):
            x, y = cell
            if 0 <= x < w and 0 <= y < h:
                grid[plane * cells + y * w + x] = 1.0

        for i in range(self.num_players):
            # Dead snakes are off the board -- `step` stops updating their
            # bodies and `_obstacles` ignores them, so the planes must too.
            if not state.alive[i]:
                continue
            body = state.bodies[i]
            head_plane, body_plane, tail_plane = (0, 1, 2) if i == player else (3, 4, 5)
            mark(head_plane, body[0])
            for cell in body[1:]:
                mark(body_plane, cell)
            # The tail gets its own plane because it is the one occupied cell
            # that is usually safe to enter: it moves out of the way, unless
            # that snake eats this tick.
            mark(tail_plane, body[-1])

        for cell in state.food:
            mark(6, cell)

        # An all-ones plane over the playable area. Convolutions pad with
        # zeros, so without this a cell past the edge looks exactly like an
        # empty one; with it, the padding itself is what marks the wall.
        base = 7 * cells
        for k in range(cells):
            grid[base + k] = 1.0
        return grid

    def _flat(self, state, player):
        body = state.bodies[player]
        head = body[0]
        span = self.width + self.height
        blocked = self._obstacles(state, skip_tail_of=player)

        danger, room = [], []
        for action in range(4):
            dx, dy = _DELTA[action]
            cell = (head[0] + dx, head[1] + dy)
            unsafe = not self._in_bounds(cell) or cell in blocked
            danger.append(1.0 if unsafe else 0.0)
            room.append(0.0 if unsafe else self._flood(cell, blocked) / 60.0)

        facing = [1.0 if state.directions[player] == a else 0.0 for a in range(4)]

        if state.food:
            fx, fy = min(
                state.food, key=lambda f: abs(head[0] - f[0]) + abs(head[1] - f[1])
            )
            food_vec = [(fx - head[0]) / self.width, (fy - head[1]) / self.height]
        else:
            food_vec = [0.0, 0.0]

        rivals = [
            state.bodies[i][0]
            for i in range(self.num_players)
            if i != player and state.alive[i]
        ]
        if rivals:
            rx, ry = min(
                rivals, key=lambda r: abs(head[0] - r[0]) + abs(head[1] - r[1])
            )
            rival_vec = [
                (rx - head[0]) / self.width,
                (ry - head[1]) / self.height,
                1.0 - (abs(head[0] - rx) + abs(head[1] - ry)) / span,
            ]
        else:
            rival_vec = [0.0, 0.0, 0.0]

        tail = [
            len(body) / (self.width * self.height),
            state.hunger[player] / self.starve_limit,
        ]
        return danger + room + facing + food_vec + rival_vec + tail

    def render(self, state):
        return {
            "kind": "snake",
            "width": self.width,
            "height": self.height,
            "snakes": [
                {"body": [list(c) for c in state.bodies[i]], "alive": state.alive[i]}
                for i in range(self.num_players)
            ],
            "food": [list(f) for f in sorted(state.food)],
            "eaten": list(state.eaten),
            "step": state.step,
        }
