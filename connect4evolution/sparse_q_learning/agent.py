"""
Our Q-Learning agent, ready to learn and play Connect Four!

This is where all the magic happens. Like a curious child playing their
first game, our agent starts knowing nothing but gradually learns through
experience what moves work best! 🎮
"""

import random

import numpy as np

from ..environment.state import GameState
from .config import AgentConfig
from .memory import SparseQTable


class QLearningAgent:
    """
    Our learning agent that grows into a Connect Four player.

    Just like how we learned games as kids, this agent starts by trying
    things out randomly, then gradually learns what works and what doesn't.
    """

    def __init__(self, config: AgentConfig):
        """Get ready to learn!"""
        self.config = config
        self.q_table = SparseQTable()
        self.epsilon = config.initial_epsilon

    def choose_action(self, state: GameState) -> int:
        """
        Pick the next move to make.

        Sometimes we try something new (explore), and sometimes we use
        what we've learned (exploit). It's like how you might try a new
        strategy sometimes, but stick to what works other times!
        """
        valid_moves = state.get_valid_moves()

        # Time to try something new?
        if random.random() < self.epsilon:
            return random.choice(valid_moves)

        # Or use what we've learned
        q_values = [self.q_table.get_value(state, action) for action in range(7)]

        # Can't pick invalid moves!
        for i in range(7):
            if i not in valid_moves:
                q_values[i] = float("-inf")

        return int(np.argmax(q_values))

    def learn(self, state: GameState, action: int, reward: float, next_state: GameState) -> None:
        """
        Learn from what happened after our move.

        This is like reflecting on a game move. We think about what we did,
        what happened next, and how we could do better next time!
        """
        # What did we think about this move before?
        current_q = self.q_table.get_value(state, action)

        # What's the best outcome possible from here?
        next_q = (
            0
            if next_state.game_over
            else max(self.q_table.get_value(next_state, a) for a in next_state.get_valid_moves())
        )

        # Update our understanding
        new_q = current_q + self.config.learning_rate * (
            reward + self.config.discount_factor * next_q - current_q
        )

        # Remember what we learned
        self.q_table.set_value(state, action, new_q)

        # Gradually become more strategic
        self.epsilon = max(self.config.min_epsilon, self.epsilon * self.config.epsilon_decay)

    def save(self, filepath: str) -> None:
        """Save everything we've learned."""
        self.q_table.save(filepath)

    def load(self, filepath: str) -> None:
        """Load our previous knowledge."""
        self.q_table.load(filepath)
