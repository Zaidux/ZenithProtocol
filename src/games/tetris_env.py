# /src/games/tetris_env.py

import numpy as np
import random
from typing import List, Tuple
from ..utils.config import Config
from ..conceptual_knowledge_graph.ckg import ConceptualKnowledgeGraph # New Import
from ..utils.knowledge_ingestor import KnowledgeIngestor # New Import

# --- Constants from Config ---
BOARD_WIDTH = Config.TETRIS_BOARD_WIDTH
BOARD_HEIGHT = Config.TETRIS_BOARD_HEIGHT

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
    return [list(row) for row in zip(*piece[::-1])]

def get_conceptual_features(board: np.ndarray) -> np.ndarray:
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
    """A production-grade Tetris game environment for an RL agent, with CKG integration."""

    def __init__(self, ckg: ConceptualKnowledgeGraph): # CKG is now a dependency
        self.board: np.ndarray = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)
        self.current_piece: List[List[int]] | None = None
        self.current_pos: List[int] = [0, 0]
        self.next_piece: List[List[int]] | None = None
        self.score: int = 0
        self.game_over: bool = False
        self.ckg = ckg
        self.knowledge_ingestor = KnowledgeIngestor(ckg)
        self.reset()

    def _get_new_piece(self) -> None:
        self.current_piece = self.next_piece
        if self.current_piece:
            self.knowledge_ingestor.add_conceptual_info(
                'new_piece', f'Spawned new piece of type: {SHAPES.index(self.current_piece)}'
            )
        self.current_pos = [0, BOARD_WIDTH // 2 - len(self.current_piece[0]) // 2]
        self.next_piece = SHAPES[random.randint(0, NUM_SHAPES - 1)]

    def _check_collision(self, board: np.ndarray, piece: List[List[int]], pos: List[int]) -> bool:
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
            self.knowledge_ingestor.add_conceptual_info(
                f'lines_cleared_{lines_cleared}',
                f'Cleared {lines_cleared} lines, earning a bonus.',
                properties={'lines_cleared': lines_cleared}
            )
        return lines_cleared

    def step(self, action: int) -> Tuple[np.ndarray, np.ndarray, float, bool, Dict]:
        """
        Takes an action and returns the new state, conceptual impact, and game status.
        The reward is now returned from ARLC, so this function is simplified.
        """
        initial_board = self.board.copy()
        temp_piece = self.current_piece
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
        if self._check_collision(self.board, self.current_piece, self.current_pos):
            self.game_over = True
        state_before = initial_board
        state_after = self.board
        conceptual_features_before = get_conceptual_features(state_before)
        conceptual_features_after = get_conceptual_features(state_after)
        conceptual_impact = {
            "delta_gaps": conceptual_features_before[0] - conceptual_features_after[0],
            "delta_height": conceptual_features_before[1] - conceptual_features_after[1],
            "lines_cleared": lines_cleared
        }
        return state_before, state_after, 0.0, self.game_over, conceptual_impact

    def reset(self) -> np.ndarray:
        self.board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)
        self.score = 0
        self.game_over = False
        self.next_piece = SHAPES[random.randint(0, NUM_SHAPES - 1)]
        self._get_new_piece()
        self.knowledge_ingestor.add_conceptual_info('game_start', 'Initialized a new game.')
        return self.board

    def get_board_state(self) -> np.ndarray:
        return self.board.copy()

    def get_current_piece_id(self) -> int:
        return SHAPES.index(self.current_piece)

    def get_next_piece_id(self) -> int:
        return SHAPES.index(self.next_piece)

    def get_conceptual_features(self) -> np.ndarray:
        return get_conceptual_features(self.board)
