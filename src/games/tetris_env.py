# /src/games/tetris_env.py

import numpy as np
import random
import pygame
from typing import List, Tuple

# --- Game Constants ---
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
BLOCK_SIZE = 30
SHAPES = [
    [[1, 1], [1, 1]],        # Square (0)
    [[1, 1, 1, 1]],          # Line (1)
    [[0, 1, 0], [1, 1, 1]],  # T-shape (2)
    [[1, 1, 0], [0, 1, 1]],  # S-shape (3)
    [[0, 1, 1], [1, 1, 0]],  # Z-shape (4)
    [[1, 0, 0], [1, 1, 1]],  # L-shape (5)
    [[0, 0, 1], [1, 1, 1]],  # J-shape (6)
]
NUM_SHAPES = len(SHAPES)
COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (100, 100, 100)]

class TetrisEnvironment:
    """A full Tetris game environment with rendering capabilities."""

    def __init__(self):
        self.board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)
        self.score = 0
        self.game_over = False
        self.current_piece = None
        self.current_pos = [0, 0]

    def _get_new_piece(self):
        """Spawns a new random piece."""
        piece_id = random.randint(0, NUM_SHAPES - 1)
        self.current_piece = SHAPES[piece_id]
        self.current_pos = [0, BOARD_WIDTH // 2 - len(self.current_piece[0]) // 2]
        if self._check_collision(self.board, self.current_piece, self.current_pos):
            self.game_over = True
            return False
        return True

    def _check_collision(self, board, piece, pos):
        """Checks if a piece at a given position collides with the board boundaries or other pieces."""
        piece_height, piece_width = len(piece), len(piece[0])
        for y in range(piece_height):
            for x in range(piece_width):
                if piece[y][x] == 1:
                    board_y, board_x = pos[0] + y, pos[1] + x
                    if (board_x < 0 or board_x >= BOARD_WIDTH or
                            board_y < 0 or board_y >= BOARD_HEIGHT or
                            board[board_y, board_x] == 1):
                        return True
        return False

    def _lock_piece(self):
        """Locks the current piece in place and checks for cleared lines."""
        piece_height, piece_width = len(self.current_piece), len(self.current_piece[0])
        for y in range(piece_height):
            for x in range(piece_width):
                if self.current_piece[y][x] == 1:
                    self.board[self.current_pos[0] + y, self.current_pos[1] + x] = 1
        
        lines_cleared = 0
        new_board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)
        new_row = BOARD_HEIGHT - 1
        for row in range(BOARD_HEIGHT - 1, -1, -1):
            if not np.all(self.board[row] == 1):
                new_board[new_row] = self.board[row]
                new_row -= 1
            else:
                lines_cleared += 1
        self.board = new_board
        self.score += lines_cleared ** 2

    def step(self, action: int):
        """
        Takes an action and updates the environment.
        Action is the x-position to place the piece.
        """
        temp_piece = self.current_piece
        temp_pos = self.current_pos[:]
        temp_pos[1] = action  # Update x-position

        # Find the drop position (y-position)
        drop_pos = temp_pos[:]
        while not self._check_collision(self.board, temp_piece, [drop_pos[0] + 1, drop_pos[1]]):
            drop_pos[0] += 1
        
        self.current_pos = drop_pos

        if not self.game_over:
            self._lock_piece()
            self._get_new_piece()

        return self.board, self.score, self.game_over

    def get_board_state(self) -> np.ndarray:
        """Returns the current board state as a NumPy array."""
        return self.board.copy()

    def get_next_piece_id(self) -> int:
        """Returns the ID of the current piece (as the 'next' piece to be placed)."""
        return SHAPES.index(self.current_piece)

    def reset(self):
        """Resets the game to its initial state."""
        self.board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)
        self.score = 0
        self.game_over = False
        self.current_piece = None
        self._get_new_piece()

