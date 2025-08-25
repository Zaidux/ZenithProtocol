# /src/games/chess_env.py

import chess
import numpy as np
import torch
from typing import Dict, List, Tuple

class ChessEnvironment:
    """A full chess game environment with feature extraction."""
    def __init__(self):
        self.board = chess.Board()

    def reset(self) -> torch.Tensor:
        """Resets the board to the starting position and returns the new state."""
        self.board.reset()
        return self.get_current_state()

    def get_current_state(self) -> torch.Tensor:
        """Returns a numeric representation of the current board for the model."""
        return self._board_to_tensor()

    def is_game_over(self) -> bool:
        """Checks if the game has ended."""
        return self.board.is_game_over()

    def get_legal_moves(self) -> List[chess.Move]:
        """Returns a list of legal moves in the current position."""
        return list(self.board.legal_moves)

    def push_move(self, move: chess.Move):
        """Executes a move on the board."""
        self.board.push(move)

    def _board_to_tensor(self) -> torch.Tensor:
        """
        Converts the chess board to a tensor representation for the neural network.
        This now matches the robust format used by the data generator.
        """
        planes = np.zeros(shape=(14, 8, 8), dtype=np.float32)
        for piece_type in chess.PIECE_TYPES:
            for square in self.board.pieces(piece_type, chess.WHITE):
                planes[piece_type - 1][chess.square_rank(square)][chess.square_file(square)] = 1
            for square in self.board.pieces(piece_type, chess.BLACK):
                planes[piece_type + 5][chess.square_rank(square)][chess.square_file(square)] = 1
        
        if self.board.has_kingside_castling_rights(chess.WHITE): planes[12][0][0] = 1
        if self.board.has_queenside_castling_rights(chess.WHITE): planes[12][0][7] = 1
        if self.board.has_kingside_castling_rights(chess.BLACK): planes[12][7][0] = 1
        if self.board.has_queenside_castling_rights(chess.BLACK): planes[12][7][7] = 1
        
        if self.board.turn == chess.WHITE: planes[13][0][0] = 1
        else: planes[13][7][7] = 1
        
        return torch.from_numpy(planes)

    def evaluate_conceptual_features(self) -> np.ndarray:
        """
        Calculates and returns a set of conceptual features for the current board state.
        These metrics are essential for the ARLC's scoring function.
        """
        return get_conceptual_features(self.board)

def get_conceptual_features(board: chess.Board) -> np.ndarray:
    """
    Calculates conceptual features for the given chess board state.
    This function is now external to the class for reusability.
    """
    piece_values = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
    }
    
    white_value = sum(piece_values[p.piece_type] for p in board.piece_map().values() if p.color == chess.WHITE)
    black_value = sum(piece_values[p.piece_type] for p in board.piece_map().values() if p.color == chess.BLACK)
    material_advantage = white_value - black_value

    white_king_sq = board.king(chess.WHITE)
    black_king_sq = board.king(chess.BLACK)
    white_safety = len(list(board.attackers(chess.BLACK, white_king_sq))) if white_king_sq else 0
    black_safety = len(list(board.attackers(chess.WHITE, black_king_sq))) if black_king_sq else 0

    center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
    white_control = sum(1 for sq in center_squares if board.is_attacked_by(chess.WHITE, sq))
    black_control = sum(1 for sq in center_squares if board.is_attacked_by(chess.BLACK, sq))

    return np.array([
        material_advantage,
        white_safety,
        black_safety,
        white_control,
        black_control
    ], dtype=np.float32)
