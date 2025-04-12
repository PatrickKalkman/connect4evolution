"""
Our training playground where our AI learns to play Connect Four! 🎮

This is where the magic of learning happens. Like a patient teacher working
with an eager student, we'll guide our AI through countless games, celebrating
its victories and learning from its mistakes.
"""

import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from connect4evolution.environment.board import ConnectFourBoard
from connect4evolution.environment.player import Player
from connect4evolution.environment.renderer import ConnectFourRenderer
from connect4evolution.environment.state import GameState
from connect4evolution.sparse_q_learning.agent import QLearningAgent
from connect4evolution.sparse_q_learning.config import AgentConfig


@dataclass
class TrainingConfig:
    """Our recipe for a perfect learning environment."""

    episodes: int = 10000  # Number of games to play
    render_every: int = 1000  # How often to show the game visually
    save_every: int = 5000  # How often to save our progress
    eval_every: int = 1000  # How often to test our skills
    model_path: str = "models/sparse_q_learning.json"
    render_delay: float = 0.3  # Seconds between moves when rendering

    # Rewards to guide our learning
    win_reward: float = 1.0
    lose_reward: float = -1.0
    draw_reward: float = 0.1
    invalid_move_reward: float = -0.5


class Trainer:
    """
    Our training supervisor that guides the learning process.

    Think of this as a friendly coach who:
    * Sets up practice games
    * Keeps track of progress
    * Celebrates improvements
    * Saves what we've learned
    """

    def __init__(self, config: TrainingConfig):
        """Get ready for a learning adventure!"""
        self.config = config
        self.board = ConnectFourBoard()
        self.renderer = ConnectFourRenderer()

        # Create our eager student
        agent_config = AgentConfig()
        self.agent = QLearningAgent(agent_config)

        # Keep track of our progress
        self.wins = []
        self.draws = []
        self.episode_lengths = []

    def train(self, progress_callback: Optional[Callable[[], None]] = None) -> None:
        """Begin our learning journey!"""
        print("🎮 Starting training! Let's watch our AI grow...")

        for episode in range(self.config.episodes):
            # Time for a new game
            self.board.reset()
            moves_made = 0
            rendering = episode % self.config.render_every == 0

            while not self.board.get_state().game_over:
                state = self.board.get_state()

                if state.current_player == Player.PLAYER_1:
                    # Our agent's turn
                    action = self.agent.choose_action(state)
                    valid_move = self.board.make_move(action)

                    # Learn from what happened
                    next_state = self.board.get_state()
                    reward = self._calculate_reward(valid_move, next_state)
                    self.agent.learn(state, action, reward, next_state)
                else:
                    # Random opponent's turn
                    valid_moves = state.get_valid_moves()
                    action = np.random.choice(valid_moves)
                    self.board.make_move(action)

                moves_made += 1

                # Show the game if it's time
                if rendering:
                    self.renderer.render(self.board.get_state())
                    time.sleep(self.config.render_delay)

            # Game over! Let's record what happened
            final_state = self.board.get_state()
            if final_state.winner == Player.PLAYER_1:
                self.wins.append(1)
            else:
                self.wins.append(0)
            self.draws.append(1 if not final_state.winner else 0)
            self.episode_lengths.append(moves_made)

            # Update progress if we have a callback
            if progress_callback:
                progress_callback()

            # Share our progress
            if (episode + 1) % 100 == 0:
                recent_wins = np.mean(self.wins[-100:])
                recent_draws = np.mean(self.draws[-100:])
                print(f"\nEpisode {episode + 1}")
                print(f"🎯 Win rate: {recent_wins:.2%}")
                print(f"🤝 Draw rate: {recent_draws:.2%}")
                print(f"📊 Average game length: {np.mean(self.episode_lengths[-100:]):.1f} moves")
                print(f"🎲 Exploration rate: {self.agent.epsilon:.2%}")

            # Time to save our progress?
            if (episode + 1) % self.config.save_every == 0:
                self.agent.save(self.config.model_path)
                print(f"\n💾 Saved model to {self.config.model_path}")

    def _calculate_reward(self, valid_move: bool, state: GameState) -> float:
        """Figure out how good (or bad) our last move was."""
        if not valid_move:
            return self.config.invalid_move_reward

        if state.game_over:
            if state.winner == Player.PLAYER_1:
                return self.config.win_reward
            elif state.winner == Player.PLAYER_2:
                return self.config.lose_reward
            return self.config.draw_reward

        return 0.0  # The game continues...
