"""Connect Four as a two-player turn-based Game.

The point of having this on the bench is that it is nothing like 2048 or Snake:
players alternate rather than move together, and **the score is zero and stays
zero until the last move**, when it becomes +1 / -1 / 0. Every intermediate
position is worth exactly nothing, so there is no dense signal to ride toward a
good policy — a GA sees only the outcome, and a DQN has to carry a single
terminal reward back across forty plies.

Boards are flat 42-tuples in row-major order, row 0 at the top, so a piece
dropped in column c lands in the largest empty row of that column. 0 is empty,
1 and 2 name the player who owns the cell.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.game import TurnBasedGame

WIDTH = 7
HEIGHT = 6
CELLS = WIDTH * HEIGHT
CONNECT = 4

EMPTY_BOARD = (0,) * CELLS

# All four directions a line can run. The reverse of each is covered by
# scanning from every cell, so right/down/down-right/down-left is complete.
_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


def _index(row, col):
    return row * WIDTH + col


def _drop_row(board, col):
    """Lowest empty row in `col`, or None if the column is full."""
    for row in range(HEIGHT - 1, -1, -1):
        if not board[_index(row, col)]:
            return row
    return None


def drop(board, col, piece):
    """Place `piece` in `col`. Returns (new_board, row); row is None if full."""
    row = _drop_row(board, col)
    if row is None:
        return board, None
    cells = list(board)
    cells[_index(row, col)] = piece
    return tuple(cells), row


def _line_from(row, col, drow, dcol):
    """The `CONNECT` cells starting at (row, col), or None if off the board."""
    cells = []
    for k in range(CONNECT):
        r, c = row + drow * k, col + dcol * k
        if not (0 <= r < HEIGHT and 0 <= c < WIDTH):
            return None
        cells.append(_index(r, c))
    return tuple(cells)


# Every window of four in play, precomputed once, so the scans below are flat
# sweeps rather than nested loops with bounds checks on every cell.
_WINDOWS = tuple(
    window
    for row in range(HEIGHT)
    for col in range(WIDTH)
    for drow, dcol in _DIRECTIONS
    if (window := _line_from(row, col, drow, dcol)) is not None
)

# Windows touching each cell, for asking "did this move win?" without a full scan.
_WINDOWS_AT = tuple(
    tuple(w for w in _WINDOWS if cell in w) for cell in range(CELLS)
)


def winning_line(board, cell):
    """The four-in-a-row through `cell`, or None. `cell` is the last move."""
    piece = board[cell]
    if not piece:
        return None
    for window in _WINDOWS_AT[cell]:
        if all(board[i] == piece for i in window):
            return window
    return None


def render_text(board):
    glyph = {0: ".", 1: "X", 2: "O"}
    return "\n".join(
        " ".join(glyph[board[_index(r, c)]] for c in range(WIDTH))
        for r in range(HEIGHT)
    )


# --- features -------------------------------------------------------------

def _threats(board, piece):
    """Count windows holding `k` of our pieces and no opponent piece, for k=2,3.

    An open three is one move from winning and an open two is the material a
    three is built from, so these two counts are most of what a linear
    evaluation of this game can express.
    """
    twos = threes = 0
    for window in _WINDOWS:
        mine = other = 0
        for i in window:
            value = board[i]
            if value == piece:
                mine += 1
            elif value:
                other += 1
        if other:
            continue
        if mine == 3:
            threes += 1
        elif mine == 2:
            twos += 1
    return twos, threes


def _immediate_wins(board, piece):
    """Columns where `piece` would win by dropping right now."""
    wins = []
    for col in range(WIDTH):
        new_board, row = drop(board, col, piece)
        if row is not None and winning_line(new_board, _index(row, col)):
            wins.append(col)
    return wins


@dataclass(frozen=True)
class Connect4State:
    board: tuple
    to_move: int          # seat index, 0 or 1
    last: int | None      # cell index of the last move, for rendering
    winner: int | None    # seat index, or None for "no winner yet or a draw"
    moves: int


class Connect4Game(TurnBasedGame):
    name = "connect4"
    num_players = 2
    action_names = tuple(f"col{c}" for c in range(WIDTH))
    feature_names = (
        "wins_now",        # this drop makes four in a row
        "blocks_win",      # this drop takes the square that would have lost us the game
        "gifts_win",       # it stacks under a winning square for the opponent
        "centre",          # closeness of the column to the middle file
        "own_threes",      # our open threes after the drop
        "own_twos",        # our open twos after the drop
        "enemy_threes",    # opponent open threes left standing after the drop
    )
    # Two 42-cell planes -- ours and theirs, always from the mover's point of
    # view so the network never has to learn the colours separately -- plus one
    # flag per column for "still playable".
    obs_size = CELLS * 2 + WIDTH

    # --- dynamics ---------------------------------------------------------

    def reset(self, rng):
        return Connect4State(EMPTY_BOARD, to_move=0, last=None, winner=None, moves=0)

    def current_player(self, state):
        return state.to_move

    def moves(self, state):
        return [c for c in range(WIDTH) if _drop_row(state.board, c) is not None]

    def play(self, state, player, action, rng):
        board, row = drop(state.board, action, self._piece(player))
        cell = _index(row, action)
        return Connect4State(
            board=board,
            to_move=1 - player,
            last=cell,
            winner=player if winning_line(board, cell) else None,
            moves=state.moves + 1,
        )

    def is_terminal(self, state):
        return state.winner is not None or state.moves >= CELLS

    def scores(self, state):
        """+1 to the winner, -1 to the loser, 0 each while unresolved or drawn.

        Zero-sum and terminal-only. A draw leaves both on 0, which `winners`
        reports as a tie and the arena splits half a win each way.
        """
        if state.winner is None:
            return (0.0, 0.0)
        return tuple(1.0 if i == state.winner else -1.0 for i in range(2))

    @staticmethod
    def _piece(player):
        return player + 1

    # --- features ---------------------------------------------------------

    def action_features(self, state, player, action):
        mine, theirs = self._piece(player), self._piece(1 - player)
        board, row = drop(state.board, action, mine)
        if row is None:
            # Not reachable through `legal_actions`, but a full column has no
            # meaningful features and must never look attractive.
            return (0.0,) * len(self.feature_names)

        cell = _index(row, action)
        if winning_line(board, cell):
            # Nothing after this matters; say so plainly rather than letting
            # position terms dilute a forced win.
            return (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # A square the opponent could have won on is one we have just taken.
        blocks = 1.0 if action in _immediate_wins(state.board, theirs) else 0.0
        # Our piece raises the stack by one: does that hand them the square above?
        gifts = 0.0
        if row > 0:
            above, _ = drop(board, action, theirs)
            if winning_line(above, _index(row - 1, action)):
                gifts = 1.0

        own_twos, own_threes = _threats(board, mine)
        _, enemy_threes = _threats(board, theirs)
        centre = 1.0 - abs(action - (WIDTH - 1) / 2) / ((WIDTH - 1) / 2)

        return (
            0.0,
            blocks,
            gifts,
            centre,
            own_threes / 4.0,
            own_twos / 8.0,
            enemy_threes / 4.0,
        )

    def observe(self, state, player, encoding=None):
        mine, theirs = self._piece(player), self._piece(1 - player)
        own_plane = [1.0 if v == mine else 0.0 for v in state.board]
        enemy_plane = [1.0 if v == theirs else 0.0 for v in state.board]
        playable = [
            1.0 if _drop_row(state.board, c) is not None else 0.0 for c in range(WIDTH)
        ]
        return own_plane + enemy_plane + playable

    def render(self, state):
        line = winning_line(state.board, state.last) if state.last is not None else None
        return {
            "kind": "connect4",
            "width": WIDTH,
            "height": HEIGHT,
            "cells": list(state.board),
            "last": state.last,
            "winner": state.winner,
            "line": list(line) if line else None,
            "moves": state.moves,
        }
