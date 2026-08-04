"""2048 as a one-player Game.

Boards are flat 16-tuples of *exponents*: 0 is an empty cell, n means the tile
2**n. Storing exponents rather than face values keeps the space of possible
rows small enough (16**4) to precompute every row transition, which is what
makes simulation fast enough to run a GA on top of.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from core.game import Game

SIZE = 4
CELLS = SIZE * SIZE
MAX_EXP = 15  # 2**15 = 32768, past anything reachable on a 4x4 board

LEFT, RIGHT, UP, DOWN = range(4)
EMPTY_BOARD = (0,) * CELLS
_CORNERS = (0, SIZE - 1, SIZE * (SIZE - 1), SIZE * SIZE - 1)


def _slide_row(row):
    """Collapse one row to the left. Returns (new_row, score_gained).

    Each tile takes part in at most one merge per move and merges resolve from
    the left, so (2, 2, 2, 0) becomes (4, 2, 0, 0) rather than (4, 4, 0, 0).
    """
    tiles = [t for t in row if t]
    out, score, i = [], 0, 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1] and tiles[i] < MAX_EXP:
            merged = tiles[i] + 1
            out.append(merged)
            score += 1 << merged
            i += 2
        else:
            out.append(tiles[i])
            i += 1
    out.extend([0] * (SIZE - len(out)))
    return tuple(out), score


_ROW_TABLE = {
    row: _slide_row(row)
    for row in itertools.product(range(MAX_EXP + 1), repeat=SIZE)
}


def _rows(board):
    return [board[i * SIZE:(i + 1) * SIZE] for i in range(SIZE)]


def _flatten(rows):
    return tuple(tile for row in rows for tile in row)


def _transpose(board):
    return tuple(board[c * SIZE + r] for r in range(SIZE) for c in range(SIZE))


def _reverse_rows(board):
    return _flatten([row[::-1] for row in _rows(board)])


def _move_left(board):
    out, score = [], 0
    for row in _rows(board):
        new_row, gained = _ROW_TABLE[row]
        out.append(new_row)
        score += gained
    return _flatten(out), score


def move(board, direction):
    """Apply a move without spawning. Returns (new_board, score_gained, changed)."""
    if direction == LEFT:
        new_board, score = _move_left(board)
    elif direction == RIGHT:
        new_board, score = _move_left(_reverse_rows(board))
        new_board = _reverse_rows(new_board)
    elif direction == UP:
        new_board, score = _move_left(_transpose(board))
        new_board = _transpose(new_board)
    elif direction == DOWN:
        new_board, score = _move_left(_reverse_rows(_transpose(board)))
        new_board = _transpose(_reverse_rows(new_board))
    else:
        raise ValueError(f"unknown direction: {direction!r}")
    return new_board, score, new_board != board


def spawn(board, rng):
    """Place a 2 (90%) or 4 (10%) on a uniformly chosen empty cell."""
    empties = [i for i, tile in enumerate(board) if not tile]
    if not empties:
        return board
    cells = list(board)
    cells[rng.choice(empties)] = 1 if rng.random() < 0.9 else 2
    return tuple(cells)


def max_tile(board):
    return 1 << max(board) if max(board) else 0


def render_text(board):
    return "\n".join(
        "".join(f"{(1 << t) if t else '.':>6}" for t in row) for row in _rows(board)
    )


# --- features -------------------------------------------------------------

def _lines(board):
    """All four rows followed by all four columns."""
    return _rows(board) + _rows(_transpose(board))


def _monotonicity(board):
    """Negative cost: 0 when every line is sorted, more negative otherwise.

    For each line we total the rises and the falls and charge the smaller of the
    two, so a perfectly ordered line pays nothing regardless of direction.
    """
    penalty = 0
    for line in _lines(board):
        rises = falls = 0
        for a, b in zip(line, line[1:]):
            if b > a:
                rises += b - a
            elif a > b:
                falls += a - b
        penalty += min(rises, falls)
    return -penalty


def _smoothness(board):
    penalty = 0
    for line in _lines(board):
        for a, b in zip(line, line[1:]):
            if a and b:
                penalty += abs(a - b)
    return -penalty


def _merges(board):
    return sum(
        1
        for line in _lines(board)
        for a, b in zip(line, line[1:])
        if a and a == b
    )


@dataclass(frozen=True)
class State2048:
    board: tuple
    score: int


class Game2048(Game):
    name = "2048"
    num_players = 1
    action_names = ("left", "right", "up", "down")
    # Order is load-bearing: existing trained weight files assume it.
    feature_names = (
        "score_gain",     # points scored by the move itself
        "empty",          # free cells, the main proxy for "not about to lose"
        "monotonicity",   # penalty for rows/columns that are not sorted
        "smoothness",     # penalty for large jumps between neighbours
        "max_corner",     # is the biggest tile pinned in a corner
        "max_tile",       # exponent of the largest tile
        "merges",         # adjacent equal pairs, i.e. merges available next turn
    )
    obs_size = CELLS + 2

    def reset(self, rng):
        return State2048(spawn(spawn(EMPTY_BOARD, rng), rng), 0)

    def legal_actions(self, state, player=0):
        return [d for d in range(4) if move(state.board, d)[2]]

    def step(self, state, actions, rng):
        new_board, gained, changed = move(state.board, actions[0])
        if not changed:
            raise ValueError(f"illegal move {actions[0]!r}")
        return State2048(spawn(new_board, rng), state.score + gained)

    def is_terminal(self, state):
        return not self.legal_actions(state)

    def scores(self, state):
        return (float(state.score),)

    def action_features(self, state, player, action):
        board, gained, _ = move(state.board, action)
        peak = max(board)
        return (
            math.log1p(gained),
            sum(1 for t in board if not t) / CELLS,
            _monotonicity(board) / 16.0,
            _smoothness(board) / 16.0,
            1.0 if any(board[c] == peak for c in _CORNERS) else 0.0,
            peak / 16.0,
            _merges(board) / 8.0,
        )

    def observe(self, state, player=0, encoding=None):
        peak = max(state.board) or 1
        return [t / 16.0 for t in state.board] + [
            sum(1 for t in state.board if not t) / CELLS,
            peak / 16.0,
        ]

    def render(self, state):
        return {
            "kind": "grid2048",
            "size": SIZE,
            "cells": [(1 << t) if t else 0 for t in state.board],
            "score": state.score,
            "max_tile": max_tile(state.board),
        }
