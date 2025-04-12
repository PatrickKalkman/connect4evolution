"""
Connect Four board implementation with game logic.

This module provides the core game mechanics, handling moves,
win detection, and board state management.
"""

from typing import Optional, Tuple

import numpy as np

from .player import Player
from .state import GameState


class ConnectFourBoard:
    """
    Implements the Connect Four game logic.

    Think of this as the referee of our game. It:
    * Keeps track of the game state
    * Validates moves
    * Checks for wins
    * Manages turn order
    """

    def __init__(self, rows: int = 6, cols: int = 7):
        """
        Initialize an empty Connect Four board.

        Args:
            rows: Number of rows in the board (default: 6)
            cols: Number of columns in the board (default: 7)
        """
        self.rows = rows
        self.cols = cols
        self._reset()

    def make_move(self, column: int) -> bool:
        """
        Attempt to make a move in the specified column.

        Args:
            column: The column to drop the piece in (0-based index)

        Returns:
            bool: True if the move was valid and made, False otherwise
        """
        if self._game_over or column not in self.get_valid_moves():
            return False

        # Find the lowest empty cell in the column
        for row in range(self.rows - 1, -1, -1):
            if self._board[row][column] == Player.EMPTY.value:
                self._board[row][column] = self._current_player.value
                self._last_move = (row, column)
                break

        # Check for win
        if self._check_win(self._last_move):
            self._game_over = True
            self._winner = self._current_player
        # Check for draw
        elif not self.get_valid_moves():
            self._game_over = True

        # Switch players
        self._current_player = self._current_player.next_player()
        return True

    def get_valid_moves(self) -> list[int]:
        """Return a list of columns where a piece can be dropped."""
        return [col for col, cell in enumerate(self._board[0]) if cell == Player.EMPTY.value]

    def get_state(self) -> GameState:
        """Return the current game state."""
        return GameState(
            board=self._board.copy(),
            current_player=self._current_player,
            last_move=self._last_move,
            game_over=self._game_over,
            winner=self._winner,
        )

    def reset(self) -> None:
        """Reset the board to its initial state."""
        self._reset()

    def _reset(self) -> None:
        """Internal method to reset the game state."""
        self._board = np.zeros((self.rows, self.cols), dtype=int)
        self._current_player = Player.PLAYER_1
        self._last_move = None
        self._game_over = False
        self._winner = None

    def _check_win(self, last_move: Optional[Tuple[int, int]]) -> bool:
        """
        Check if the last move resulted in a win.

        This method checks all possible winning directions from the last move made,
        which is more efficient than checking the entire board.
        """
        if not last_move:
            return False

        row, col = last_move
        player = self._board[row][col]

        # Define direction pairs to check
        directions = [
            [(0, 1), (0, -1)],  # Horizontal
            [(1, 0), (-1, 0)],  # Vertical
            [(1, 1), (-1, -1)],  # Diagonal /
            [(1, -1), (-1, 1)],  # Diagonal \
        ]

        # Check each direction pair
        for dir1, dir2 in directions:
            count = 1
            # Check first direction
            count += self._count_consecutive(row, col, dir1[0], dir1[1], player)
            # Check opposite direction
            count += self._count_consecutive(row, col, dir2[0], dir2[1], player)
            if count >= 4:
                return True
        return False

    def _count_consecutive(
        self, row: int, col: int, row_dir: int, col_dir: int, player: int
    ) -> int:
        """Count consecutive pieces in a direction."""
        count = 0
        current_row, current_col = row + row_dir, col + col_dir

        while (
            0 <= current_row < self.rows
            and 0 <= current_col < self.cols
            and self._board[current_row][current_col] == player
        ):
            count += 1
            current_row += row_dir
            current_col += col_dir

        return count
