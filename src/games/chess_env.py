# /src/games/chess_env.py

import chess
import numpy as np
import torch
from typing import Dict, List, Tuple

class ChessEnvironment:
    """A full chess game environment with feature extraction."""
    def __init__(self):
        self.board = chess.Board()

        # Piece values for material evaluation
        self.piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
        }

    def get_current_state(self) -> torch.Tensor:
        """Returns a numeric representation of the current board for the model."""
        return self._board_to_features()

    def is_game_over(self) -> bool:
        """Checks if the game has ended."""
        return self.board.is_game_over()

    def get_legal_moves(self) -> List[chess.Move]:
        """Returns a list of legal moves in the current position."""
        return list(self.board.legal_moves)

    def push_move(self, move: chess.Move):
        """Executes a move on the board."""
        self.board.push(move)

    def _board_to_features(self) -> torch.Tensor:
        """
        Converts the chess board to a tensor representation for the neural network.
        Each piece type and color is a separate channel in the tensor.
        This provides a rich "image" for the model to process.
        """
        board_tensor = torch.zeros(12, 8, 8, dtype=torch.float32)
        piece_map = self.board.piece_map()
        for sq, piece in piece_map.items():
            rank, file = chess.square_rank(sq), chess.square_file(sq)
            piece_idx = piece.piece_type - 1  # 0-5 for pawn to king
            color_offset = 6 if piece.color == chess.BLACK else 0
            board_tensor[piece_idx + color_offset, rank, file] = 1.0
        return board_tensor

    def evaluate_conceptual_features(self) -> np.ndarray:
        """
        Calculates and returns a set of conceptual features for the current board state.
        These metrics are essential for the ARLC's scoring function.
        """
        material_advantage = self._get_material_value()
        king_safety = self._get_king_safety()
        center_control = self._get_center_control()
        
        return np.array([
            material_advantage,
            king_safety[0],  # White king safety
            king_safety[1],  # Black king safety
            center_control[0], # White center control
            center_control[1]  # Black center control
        ], dtype=np.float32)

    def _get_material_value(self) -> float:
        """Calculates the material advantage for White."""
        white_value = sum(self.piece_values[p.piece_type] for p in self.board.piece_map().values() if p.color == chess.WHITE)
        black_value = sum(self.piece_values[p.piece_type] for p in self.board.piece_map().values() if p.color == chess.BLACK)
        return white_value - black_value

    def _get_king_safety(self) -> Tuple[float, float]:
        """A simple heuristic for king safety based on surrounding pieces."""
        white_king_sq = self.board.king(chess.WHITE)
        black_king_sq = self.board.king(chess.BLACK)
        white_safety, black_safety = 0.0, 0.0
        
        if white_king_sq:
            white_safety = len(list(self.board.attackers(chess.BLACK, white_king_sq)))
        if black_king_sq:
            black_safety = len(list(self.board.attackers(chess.WHITE, black_king_sq)))
            
        return white_safety, black_safety

    def _get_center_control(self) -> Tuple[int, int]:
        """Counts the number of pieces controlling the center."""
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        white_control = sum(1 for sq in center_squares if self.board.is_attacked_by(chess.WHITE, sq))
        black_control = sum(1 for sq in center_squares if self.board.is_attacked_by(chess.BLACK, sq))
        return white_control, black_control

---
