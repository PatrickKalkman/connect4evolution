"""
PyGame-based renderer for Connect Four.

This module handles all the visual aspects of our game, creating a clean,
beautiful interface that works seamlessly with any AI approach we choose.
Think of it as our game's artist, bringing the abstract game state to life! 🎨
"""

from typing import Optional, Tuple

import pygame

from .player import Player
from .state import GameState


class ConnectFourRenderer:
    """
    A beautiful, responsive Connect Four visualization.

    Features:
    * Smooth animations for token drops
    * Hover effects for showing next move
    * Win highlighting
    * Clean, modern design
    """

    # Color palette for our game
    COLORS = {
        "BACKGROUND": (41, 41, 41),  # Dark gray
        "BOARD": (35, 94, 168),  # Deep blue
        "HOVER": (69, 123, 191),  # Light blue
        "PLAYER1": (229, 57, 53),  # Red
        "PLAYER2": (253, 216, 53),  # Yellow
        "EMPTY": (200, 200, 200),  # Light gray
        "WIN_HIGHLIGHT": (46, 204, 113),  # Green
    }

    def __init__(self, rows: int = 6, cols: int = 7, cell_size: int = 80):
        """Initialize the renderer with given dimensions."""
        pygame.init()

        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size

        # Calculate window dimensions
        self.width = cols * cell_size
        self.height = (rows + 1) * cell_size  # Extra row for token preview

        # Initialize display
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Connect Four Evolution")

        # Animation state
        self.dropping_token = None
        self.drop_y = 0
        self.target_y = 0

    def render(self, state: GameState, hover_col: Optional[int] = None) -> None:
        """
        Render the current game state.

        Args:
            state: Current game state
            hover_col: Column where the mouse is hovering (for preview)
        """
        self.screen.fill(self.COLORS["BACKGROUND"])

        # Draw the board background
        pygame.draw.rect(
            self.screen,
            self.COLORS["BOARD"],
            (0, self.cell_size, self.width, self.height - self.cell_size),
        )

        # Draw hover preview
        if hover_col is not None and not state.game_over:
            self._draw_hover_preview(hover_col, state.current_player)

        # Draw all tokens
        for row in range(self.rows):
            for col in range(self.cols):
                self._draw_cell(row, col, state.board[row][col])

        # Highlight winning tokens if game is over
        if state.game_over and state.winner and state.last_move:
            self._highlight_win(state)

        pygame.display.flip()

    def handle_events(self) -> Tuple[bool, Optional[int]]:
        """
        Handle PyGame events and return user input.

        Returns:
            Tuple of (game_running, selected_column)
        """
        running = True
        selected_col = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                selected_col = x // self.cell_size

        # Get hover position
        x, _ = pygame.mouse.get_pos()
        hover_col = x // self.cell_size

        # Only render hover effect if mouse is in valid column
        if 0 <= hover_col < self.cols:
            self.hover_column = hover_col

        return running, selected_col

    def _draw_cell(self, row: int, col: int, player_value: int) -> None:
        """Draw a single cell with a token if present."""
        color = {
            Player.EMPTY.value: self.COLORS["EMPTY"],
            Player.PLAYER_1.value: self.COLORS["PLAYER1"],
            Player.PLAYER_2.value: self.COLORS["PLAYER2"],
        }[player_value]

        center = (
            col * self.cell_size + self.cell_size // 2,
            (row + 1) * self.cell_size + self.cell_size // 2,
        )

        radius = int(self.cell_size * 0.4)
        pygame.draw.circle(self.screen, color, center, radius)

    def _draw_hover_preview(self, col: int, current_player: Player) -> None:
        """Draw a preview token at the top of the selected column."""
        color = (
            self.COLORS["PLAYER1"] if current_player == Player.PLAYER_1 else self.COLORS["PLAYER2"]
        )

        center = (col * self.cell_size + self.cell_size // 2, self.cell_size // 2)

        radius = int(self.cell_size * 0.4)
        pygame.draw.circle(self.screen, color, center, radius)

    def _highlight_win(self, state: GameState) -> None:
        """Highlight the winning tokens."""
        if not state.last_move:
            return

        row, col = state.last_move
        player = state.board[row][col]

        # Define directions to check
        directions = [
            [(0, 1), (0, -1)],  # Horizontal
            [(1, 0), (-1, 0)],  # Vertical
            [(1, 1), (-1, -1)],  # Diagonal /
            [(1, -1), (-1, 1)],  # Diagonal \
        ]

        # Check each direction for a win
        for dir1, dir2 in directions:
            tokens = [(row, col)]

            # Check both directions
            for dr, dc in [dir1, dir2]:
                r, c = row + dr, col + dc
                while 0 <= r < self.rows and 0 <= c < self.cols and state.board[r][c] == player:
                    tokens.append((r, c))
                    r, c = r + dr, c + dc

            # If we found a win, highlight these tokens
            if len(tokens) >= 4:
                for r, c in tokens:
                    center = (
                        c * self.cell_size + self.cell_size // 2,
                        (r + 1) * self.cell_size + self.cell_size // 2,
                    )
                    radius = int(self.cell_size * 0.4)
                    pygame.draw.circle(
                        self.screen, self.COLORS["WIN_HIGHLIGHT"], center, radius + 4
                    )
                return

    def cleanup(self) -> None:
        """Clean up PyGame resources."""
        pygame.quit()
