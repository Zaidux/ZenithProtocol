# /src/data/chess_generator.py

import chess
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
from ..utils.config import Config

# Constants for move encoding
NUM_SQUARES = 64
NUM_MOVES = NUM_SQUARES * NUM_SQUARES # 4096 possible moves (from-square, to-square)

def get_conceptual_features(board: chess.Board) -> np.ndarray:
    """
    Calculates conceptual features for the given chess board state.
    This is the "Understanding is Key" component for the Chess domain.
    """
    piece_values = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
    }
    
    # Material Advantage
    white_value = sum(piece_values[p.piece_type] for p in board.piece_map().values() if p.color == chess.WHITE)
    black_value = sum(piece_values[p.piece_type] for p in board.piece_map().values() if p.color == chess.BLACK)
    material_advantage = white_value - black_value

    # King Safety (simple heuristic based on attacks)
    white_king_sq = board.king(chess.WHITE)
    black_king_sq = board.king(chess.BLACK)
    white_safety = len(list(board.attackers(chess.BLACK, white_king_sq))) if white_king_sq else 0
    black_safety = len(list(board.attackers(chess.WHITE, black_king_sq))) if black_king_sq else 0

    # Center Control
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

class ChessDataset(Dataset):
    """A dataset that generates chess game data on the fly."""
    def __init__(self, size=250000):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int, torch.Tensor]:
        board = chess.Board()
        # Simulate a few random moves to get a non-starting position
        for _ in range(random.randint(5, 20)):
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            board.push(random.choice(legal_moves))
            
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            # If no legal moves, reset and try again
            return self.__getitem__(random.randint(0, self.size - 1))
        
        move_to_make = random.choice(legal_moves)
        
        board_tensor_before = self._board_to_tensor(board)
        conceptual_features_before = get_conceptual_features(board)
        
        move_idx = self._move_to_index(move_to_make)

        return board_tensor_before, move_idx, torch.from_numpy(conceptual_features_before)

    def _board_to_tensor(self, board: chess.Board) -> torch.Tensor:
        """
        Converts a chess board into a tensor representation.
        We'll use a 14-channel representation (12 for pieces, 2 for turn/castling rights).
        """
        planes = np.zeros(shape=(14, 8, 8), dtype=np.float32)
        for piece_type in chess.PIECE_TYPES:
            for square in board.pieces(piece_type, chess.WHITE):
                planes[piece_type - 1][chess.square_rank(square)][chess.square_file(square)] = 1
            for square in board.pieces(piece_type, chess.BLACK):
                planes[piece_type + 5][chess.square_rank(square)][chess.square_file(square)] = 1
        
        # Castling rights
        if board.has_kingside_castling_rights(chess.WHITE): planes[12][0][0] = 1
        if board.has_queenside_castling_rights(chess.WHITE): planes[12][0][7] = 1
        if board.has_kingside_castling_rights(chess.BLACK): planes[12][7][0] = 1
        if board.has_queenside_castling_rights(chess.BLACK): planes[12][7][7] = 1
        
        # Turn
        if board.turn == chess.WHITE: planes[13][0][0] = 1
        else: planes[13][7][7] = 1
        
        return torch.from_numpy(planes)

    def _move_to_index(self, move: chess.Move) -> int:
        """
        Converts a chess move to a unique integer index (0-4095).
        """
        from_square = move.from_square
        to_square = move.to_square
        return from_square * NUM_SQUARES + to_square

