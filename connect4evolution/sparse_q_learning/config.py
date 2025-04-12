"""
Configuration settings for our Q-Learning agent.

Think of this as our agent's personality traits. Just like how we all learn
differently, these settings shape how our AI discovers and grows! 🌱
"""

from dataclasses import dataclass


@dataclass
class AgentConfig:
    """
    Configuration for our learning agent.

    These parameters are like the dials and knobs that tune how our agent learns.
    Each one plays a special role in shaping the learning journey.
    """

    learning_rate: float = 0.1  # How quickly we learn from new experiences
    discount_factor: float = 0.95  # How much we value future rewards
    initial_epsilon: float = 1.0  # Start with 100% exploration
    epsilon_decay: float = 0.995  # Gradually reduce exploration
    min_epsilon: float = 0.01  # Always maintain some exploration

    def __post_init__(self):
        """Make sure our settings make sense."""
        if not (0 <= self.learning_rate <= 1):
            raise ValueError("Learning rate must be between 0 and 1")
        if not (0 <= self.discount_factor <= 1):
            raise ValueError("Discount factor must be between 0 and 1")
