"""
Game state representation for Connect Four.

This module provides a clean, immutable representation of the game state,
making it perfect for use in our learning algorithms and game analysis.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .player import Player


@dataclass(frozen=True)
class GameState:
    """
    Represents the complete state of a Connect Four game.

    This immutable state representation includes:
    * The current board configuration
    * Whose turn it is
    * The last move made
    * Whether the game is over
    * Who won (if anyone)

    Using a frozen dataclass ensures our game states can't be accidentally modified,
    which is crucial for our learning algorithms.
    """

    board: np.ndarray
    current_player: Player
    last_move: Optional[Tuple[int, int]]  # (row, col)
    game_over: bool
    winner: Optional[Player]

    def get_valid_moves(self) -> list[int]:
        """Returns a list of valid moves (columns) for the current state."""
        return [col for col, cell in enumerate(self.board[0]) if cell == Player.EMPTY.value]

    def __str__(self) -> str:
        """
        Returns a human-readable string representation of the game state.

        This is particularly useful for debugging and logging.
        """
        board_str = "\n".join(
            [
                " ".join(
                    [
                        "⚪"
                        if cell == Player.EMPTY.value
                        else "🔴"
                        if cell == Player.PLAYER_1.value
                        else "🟡"
                        for cell in row
                    ]
                )
                for row in self.board
            ]
        )

        status = (
            "Game in progress"
            if not self.game_over
            else f"Game over - Winner: {self.winner}"
            if self.winner
            else "Game over - Draw"
        )

        return f"{board_str}\n{status}\nCurrent player: {self.current_player}"
