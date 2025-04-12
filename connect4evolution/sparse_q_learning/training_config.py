from dataclasses import dataclass


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
