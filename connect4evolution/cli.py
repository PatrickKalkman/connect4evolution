"""
Command-line interface for Connect4Evolution.

This is where we welcome users to our Connect Four AI playground,
making it easy and fun to train and play with different agents! 🎮
"""

import sys
from enum import Enum
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from connect4evolution.sparse_q_learning.trainer import Trainer, TrainingConfig

app = typer.Typer(
    help="Connect4Evolution: Reinforcement learning approaches for Connect Four",
    rich_markup_mode="rich",
)
console = Console()


class AgentType(str, Enum):
    """The different kinds of AI players we can create."""

    SPARSE_Q = "sparse_q"
    DEEP_Q = "deep_q"
    ALPHAZERO = "alphazero"


@app.command()
def play(
    agent: AgentType = typer.Option(AgentType.SPARSE_Q, help="Agent type to play against"),
    model: str = typer.Option(None, help="Path to model file"),
):
    """
    Challenge yourself against our AI! 🎮

    Pick your opponent and let's see who wins!
    """
    console.print(f"🤖 Playing against {agent.value} agent")
    if model:
        console.print(f"📚 Using model from: {model}")
    else:
        console.print("⚠️ No model specified, using default settings")

    # TODO: Implement play logic
    console.print("🚧 Play mode coming soon!")


@app.command()
def train(
    agent: AgentType = typer.Argument(..., help="Agent type to train"),
    episodes: int = typer.Option(100000, help="Number of training episodes"),
    output: str = typer.Option("models/model.json", help="Output path for trained model"),
    learning_rate: float = typer.Option(0.1, help="Learning rate for training"),
    self_play: bool = typer.Option(False, help="Whether to use self-play for training"),
):
    """
    Train an AI to master Connect Four! 📚

    Watch as your agent learns and grows through experience.
    """
    # Create output directory if it doesn't exist
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    console.print(f"\n🎓 Starting training for {agent.value} agent")
    console.print(f"📊 Planning to play {episodes} episodes")
    console.print(f"📈 Learning rate: {learning_rate}")
    console.print(f"🤝 Self-play mode: {'on' if self_play else 'off'}\n")

    if agent == AgentType.SPARSE_Q:
        # Configure our training session
        training_config = TrainingConfig(
            episodes=episodes,
            model_path=output,
            render_every=max(episodes // 20, 1000),  # Show progress regularly
            save_every=max(episodes // 10, 5000),  # Save checkpoints
        )

        # Create and run our trainer
        trainer = Trainer(training_config)

        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Training in progress...", total=episodes)
            trainer.train(progress_callback=lambda: progress.update(task, advance=1))

        console.print("\n✨ Training complete!")
        console.print(f"💾 Model saved to: {output}")

    else:
        console.print("🚧 This agent type isn't implemented yet!")


def main():
    """Welcome to Connect4Evolution! Let's play and learn together."""
    app()


if __name__ == "__main__":
    sys.exit(main())
