"""Command-line interface for Connect4Evolution."""

import sys
from enum import Enum

import typer

app = typer.Typer(help="Connect4Evolution: Reinforcement learning approaches for Connect Four")


class AgentType(str, Enum):
    SPARSE_Q = "sparse_q"
    DEEP_Q = "deep_q"
    ALPHAZERO = "alphazero"


@app.command()
def play(
    agent: AgentType = typer.Option(AgentType.ALPHAZERO, help="Agent type to play against"),
    model: str = typer.Option(None, help="Path to model file"),
):
    """Play against a trained agent."""
    typer.echo(f"Playing against {agent.value} agent with model {model}")
    # Implement play logic here


@app.command()
def train(
    agent: AgentType = typer.Argument(..., help="Agent type to train"),
    episodes: int = typer.Option(100000, help="Number of training episodes"),
    output: str = typer.Option("model.pth", help="Output path for trained model"),
    learning_rate: float = typer.Option(0.001, help="Learning rate for training"),
    self_play: bool = typer.Option(False, help="Whether to use self-play for training"),
):
    """Train an agent."""
    typer.echo(f"Training {agent.value} agent for {episodes} episodes")
    typer.echo(f"Learning rate: {learning_rate}, Self-play: {self_play}")
    # Implement training logic here


def main():
    """Run the Connect4Evolution CLI."""
    app()


if __name__ == "__main__":
    sys.exit(main())
