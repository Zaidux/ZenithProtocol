# /src/games/tetris_env.py

import numpy as np
import random
from typing import List, Tuple
from ..utils.config import Config

# --- Constants from Config ---
BOARD_WIDTH = Config.TETRIS_BOARD_WIDTH
BOARD_HEIGHT = Config.TETRIS_BOARD_HEIGHT

# Tetromino shapes (O, I, T, S, Z, L, J)
SHAPES = [
    [[1, 1], [1, 1]],        # O-shape (0)
    [[1, 1, 1, 1]],          # I-shape (1)
    [[0, 1, 0], [1, 1, 1]],  # T-shape (2)
    [[0, 1, 1], [1, 1, 0]],  # S-shape (3)
    [[1, 1, 0], [0, 1, 1]],  # Z-shape (4)
    [[1, 0, 0], [1, 1, 1]],  # L-shape (5)
    [[0, 0, 1], [1, 1, 1]],  # J-shape (6)
]
NUM_SHAPES = len(SHAPES)

def rotate_piece(piece: List[List[int]]) -> List[List[int]]:
    """Rotates a piece 90 degrees clockwise."""
    return [list(row) for row in zip(*piece[::-1])]

def get_conceptual_features(board: np.ndarray) -> np.ndarray:
    """Calculates conceptual features of a board state."""
    gaps = 0
    for col in range(board.shape[1]):
        filled_found = False
        for row in range(board.shape[0]):
            if board[row, col] == 1:
                filled_found = True
            elif filled_found:
                gaps += 1

    max_height = 0
    for col in range(board.shape[1]):
        for row in range(board.shape[0]):
            if board[row, col] == 1:
                col_height = BOARD_HEIGHT - row
                max_height = max(max_height, col_height)
                break

    board_fullness = np.sum(board) / (BOARD_WIDTH * BOARD_HEIGHT)

    return np.array([gaps, max_height, board_fullness], dtype=np.float32)

class TetrisEnvironment:
    """A production-grade Tetris game environment for an RL agent."""

    def __init__(self):
        self.board: np.ndarray = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)
        self.current_piece: List[List[int]] | None = None
        self.current_pos: List[int] = [0, 0]
        self.next_piece: List[List[int]] | None = None
        self.score: int = 0
        self.game_over: bool = False
        self.reset()

    def _get_new_piece(self) -> None:
        """Spawns a new random piece."""
        self.current_piece = self.next_piece
        self.current_pos = [0, BOARD_WIDTH // 2 - len(self.current_piece[0]) // 2]
        self.next_piece = SHAPES[random.randint(0, NUM_SHAPES - 1)]

    def _check_collision(self, board: np.ndarray, piece: List[List[int]], pos: List[int]) -> bool:
        """Checks if a piece at a given position collides with the board."""
        piece_height, piece_width = len(piece), len(piece[0])
        for y in range(piece_height):
            for x in range(piece_width):
                if piece[y][x] == 1:
                    board_y, board_x = pos[0] + y, pos[1] + x
                    if (board_x < 0 or board_x >= BOARD_WIDTH or
                            board_y >= BOARD_HEIGHT or
                            board[board_y, board_x] == 1):
                        return True
        return False

    def _lock_piece(self) -> int:
        """Locks the current piece in place and checks for cleared lines."""
        lines_cleared = 0
        piece_height, piece_width = len(self.current_piece), len(self.current_piece[0])
        for y in range(piece_height):
            for x in range(piece_width):
                if self.current_piece[y][x] == 1:
                    self.board[self.current_pos[0] + y, self.current_pos[1] + x] = 1

        lines_to_clear = [i for i, row in enumerate(self.board) if np.all(row == 1)]
        if lines_to_clear:
            lines_cleared = len(lines_to_clear)
            new_board = np.zeros_like(self.board)
            rows_to_keep = [i for i in range(BOARD_HEIGHT) if i not in lines_to_clear]
            new_board[lines_cleared:] = self.board[rows_to_keep]
            self.board = new_board
        
        return lines_cleared

    def step(self, action: int) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        """
        Takes an action (move) and returns the new state, reward, and game over status.
        Action is the x-position to place the piece.
        """
        initial_board = self.board.copy()
        temp_piece = self.current_piece
        
        # Validate and apply action (x-position)
        x_pos = max(0, min(action, BOARD_WIDTH - len(temp_piece[0])))
        y_pos = 0
        while not self._check_collision(self.board, temp_piece, [y_pos + 1, x_pos]):
            y_pos += 1
            if y_pos + len(temp_piece) > BOARD_HEIGHT:
                break
        
        self.current_pos = [y_pos, x_pos]
        
        if self._check_collision(self.board, temp_piece, self.current_pos):
            self.game_over = True
        
        if not self.game_over:
            lines_cleared = self._lock_piece()
            self.score += lines_cleared
            self._get_new_piece()

        # Check for game over on next piece
        if self._check_collision(self.board, self.current_piece, self.current_pos):
            self.game_over = True

        reward = self._calculate_reward(initial_board)

        state_before = initial_board
        state_after = self.board
        
        return state_before, state_after, reward, self.game_over

    def _calculate_reward(self, board_before: np.ndarray) -> float:
        """Calculates a heuristic reward based on changes in conceptual features."""
        features_before = get_conceptual_features(board_before)
        features_after = get_conceptual_features(self.board)
        
        delta_gaps = features_before[0] - features_after[0]
        delta_height = features_before[1] - features_after[1]
        
        reward = (Config.ARLC_REWARD_COEFFS['tetris']['lines_cleared'] * self.score) + \
                 (Config.ARLC_REWARD_COEFFS['tetris']['gaps'] * delta_gaps) + \
                 (Config.ARLC_REWARD_COEFFS['tetris']['max_height'] * delta_height)
        
        return float(reward)

    def reset(self) -> np.ndarray:
        """Resets the game to its initial state."""
        self.board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)
        self.score = 0
        self.game_over = False
        self.next_piece = SHAPES[random.randint(0, NUM_SHAPES - 1)]
        self._get_new_piece()
        return self.board

    def get_board_state(self) -> np.ndarray:
        """Returns the current board state as a NumPy array."""
        return self.board.copy()

    def get_current_piece(self) -> List[List[int]]:
        """Returns the shape of the current piece."""
        return self.current_piece

    def get_current_piece_id(self) -> int:
        """Returns the ID of the current piece."""
        return SHAPES.index(self.current_piece)

    def get_next_piece_id(self) -> int:
        """Returns the ID of the next piece."""
        return SHAPES.index(self.next_piece)
