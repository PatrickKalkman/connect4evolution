"""
Player representation for Connect Four game.

This module defines the possible states a cell can have in the game board
and provides a clean enumeration for player turns.
"""

from enum import Enum


class Player(Enum):
    """
    Enumeration of possible cell states in the game.

    Think of this as the three possible things you might see in any space:
    * An empty cell (EMPTY)
    * A red token (PLAYER_1)
    * A yellow token (PLAYER_2)
    """

    EMPTY = 0
    PLAYER_1 = 1
    PLAYER_2 = 2

    def next_player(self) -> "Player":
        """Get the next player in turn sequence."""
        if self == Player.PLAYER_1:
            return Player.PLAYER_2
        elif self == Player.PLAYER_2:
            return Player.PLAYER_1
        return self  # For EMPTY, return itself

    def __str__(self) -> str:
        """Human-readable representation of the player."""
        return {Player.EMPTY: "Empty", Player.PLAYER_1: "Player 1", Player.PLAYER_2: "Player 2"}[
            self
        ]
