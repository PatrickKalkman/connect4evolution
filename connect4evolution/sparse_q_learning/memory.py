"""
Our agent's memory system, storing and recalling game experiences.

Just like how we remember important moments in a game, this system helps
our agent keep track of what it's learned about different game situations! 💭
"""

import json
from typing import Dict

import numpy as np

from ..environment.state import GameState


class SparseQTable:
    """
    A clever way to store what our agent learns.

    Instead of trying to remember every possible game situation (which would
    be like trying to memorize every grain of sand on a beach!), we only
    store the situations we've actually seen.
    """

    def __init__(self):
        """Start with a fresh memory."""
        self._q_values: Dict[str, np.ndarray] = {}
        self._default_value = 0.0

    def get_value(self, state: GameState, action: int) -> float:
        """
        Look up how good we think a move is in a given situation.

        Like checking our notes about a similar game we played before!
        """
        state_key = self._get_state_key(state)
        if state_key not in self._q_values:
            self._q_values[state_key] = np.full(7, self._default_value)
        return self._q_values[state_key][action]

    def set_value(self, state: GameState, action: int, value: float) -> None:
        """
        Update our memory about how good a move is.

        This is like writing in our game journal after we learn
        something new!
        """
        state_key = self._get_state_key(state)
        if state_key not in self._q_values:
            self._q_values[state_key] = np.full(7, self._default_value)
        self._q_values[state_key][action] = value

    def _get_state_key(self, state: GameState) -> str:
        """
        Create a unique 'snapshot' of the game situation.

        This is like taking a picture of the game board that we can
        use to find our notes about similar situations later.
        """
        return f"{state.board.tobytes()}{state.current_player.value}"

    def save(self, filepath: str) -> None:
        """Save all our memories to a file for later."""
        serializable = {key: values.tolist() for key, values in self._q_values.items()}
        with open(filepath, "w") as f:
            json.dump(serializable, f)

    def load(self, filepath: str) -> None:
        """Load our saved memories."""
        with open(filepath, "r") as f:
            serializable = json.load(f)
        self._q_values = {key: np.array(values) for key, values in serializable.items()}
