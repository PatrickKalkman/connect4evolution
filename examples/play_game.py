"""
A simple example showing how to play Connect Four with our implementation.

This script creates a human vs human game, showcasing our board logic
and visualization working together seamlessly! 🎮
"""

import os
import sys
import time

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connect4evolution.environment.board import ConnectFourBoard
from connect4evolution.environment.player import Player
from connect4evolution.environment.renderer import ConnectFourRenderer


def main():
    """Run a human vs human Connect Four game."""
    print("🎮 Welcome to Connect Four! Red goes first, Yellow second.")
    print("Click on any column to drop your token. Press ESC or close the window to quit.")

    # Create our game pieces
    board = ConnectFourBoard()
    renderer = ConnectFourRenderer()

    # Game loop
    running = True
    while running:
        # Get the current game state
        state = board.get_state()

        # Handle player input and update display
        running, selected_col = renderer.handle_events()

        # Make a move if a column was selected
        if selected_col is not None:
            if selected_col in state.get_valid_moves():
                board.make_move(selected_col)
                # Add satisfying sound effect or visual feedback here later!
            else:
                print("⚠️ That column is full! Try another one.")

        # Draw everything
        renderer.render(state)

        # Check for game over
        if state.game_over:
            if state.winner:
                winner = "Red 🔴" if state.winner == Player.PLAYER_1 else "Yellow 🟡"
                print(f"\n🎉 Game Over! {winner} wins!")
            else:
                print("\n🤝 Game Over! It's a draw!")

            # Keep the final state visible for a moment
            time.sleep(2)
            running = False

    # Clean up
    renderer.cleanup()


if __name__ == "__main__":
    main()
