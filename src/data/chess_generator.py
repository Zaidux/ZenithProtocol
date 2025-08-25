# /src/data/chess_generator.py

import chess
import chess.engine
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Tuple

# You'll need to install the python-chess library:
# pip install python-chess

class ChessDataGenerator:
    """
    Generates chess data by simulating games and extracting conceptual features.
    """
    def __init__(self):
        self.board = chess.Board()

        # Piece values for material evaluation
        self.piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
        }

    def _get_material_value(self, board: chess.Board) -> float:
        """Calculates the material advantage for White."""
        white_value = sum(self.piece_values[p.piece_type] for p in board.piece_map().values() if p.color == chess.WHITE)
        black_value = sum(self.piece_values[p.piece_type] for p in board.piece_map().values() if p.color == chess.BLACK)
        return white_value - black_value

    def _get_king_safety(self, board: chess.Board) -> Tuple[float, float]:
        """A simple heuristic for king safety based on surrounding pieces."""
        white_king_sq = board.king(chess.WHITE)
        black_king_sq = board.king(chess.BLACK)
        white_safety, black_safety = 0.0, 0.0
        
        if white_king_sq:
            white_safety = len(list(board.attackers(chess.BLACK, white_king_sq)))
        if black_king_sq:
            black_safety = len(list(board.attackers(chess.WHITE, black_king_sq)))
            
        return white_safety, black_safety

    def _get_center_control(self, board: chess.Board) -> Tuple[int, int]:
        """Counts the number of pieces controlling the center."""
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        white_control = sum(1 for sq in center_squares if board.is_attacked_by(chess.WHITE, sq))
        black_control = sum(1 for sq in center_squares if board.is_attacked_by(chess.BLACK, sq))
        return white_control, black_control

    def evaluate_conceptual_features(self, board: chess.Board) -> np.ndarray:
        """
        Calculates all conceptual features for the given board state.
        This function is the heart of the "Understanding is Key" principle for chess.
        """
        material_advantage = self._get_material_value(board)
        white_king_safety, black_king_safety = self._get_king_safety(board)
        white_center_control, black_center_control = self._get_center_control(board)
        
        return np.array([
            material_advantage,
            white_king_safety,
            black_king_safety,
            white_center_control,
            black_center_control
        ], dtype=np.float32)

    def generate_data_pair(self) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """
        Generates a training data pair: board state, a random legal move,
        and the resulting conceptual features.
        """
        board_before = chess.Board()
        # Simulate a few random moves to get a non-starting position
        for _ in range(random.randint(1, 10)):
            legal_moves = list(board_before.legal_moves)
            if not legal_moves:
                break
            board_before.push(random.choice(legal_moves))
        
        legal_moves = list(board_before.legal_moves)
        if not legal_moves:
            return None, None, None
            
        move_to_make = random.choice(legal_moves)
        board_after = board_before.copy()
        board_after.push(move_to_make)
        
        # Convert the board state to a tensor format for the model
        board_tensor = self._board_to_tensor(board_before)
        
        # Calculate the conceptual features for the resulting board state
        conceptual_features = self.evaluate_conceptual_features(board_after)
        
        return board_tensor, self._move_to_index(move_to_make), conceptual_features

    def _board_to_tensor(self, board: chess.Board) -> torch.Tensor:
        """
        Converts a chess board into a tensor representation.
        We'll use a simple one-hot encoding for each square and piece type.
        """
        board_tensor = torch.zeros(12, 8, 8, dtype=torch.float32)
        piece_map = board.piece_map()
        for sq, piece in piece_map.items():
            rank, file = chess.square_rank(sq), chess.square_file(sq)
            piece_idx = piece.piece_type - 1  # 0-5 for pawn to king
            color_offset = 6 if piece.color == chess.BLACK else 0
            board_tensor[piece_idx + color_offset, rank, file] = 1.0
        return board_tensor

    def _move_to_index(self, move: chess.Move) -> int:
        """
        Converts a move to a unique integer index. This is a simplification
        and would be more complex in a full engine. For now, we'll just use
        the 'to' square as an index.
        """
        # This is a highly simplified representation and needs to be expanded
        # to a full move encoding scheme for a robust model.
        return move.to_square
        
class ChessDataset(Dataset):
    """A dataset that generates chess game data on the fly."""
    def __init__(self, size=250000):
        self.size = size
        self.data_generator = ChessDataGenerator()

    def __len__(self):
        return self.size

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        board_tensor, move_idx, conceptual_features = self.data_generator.generate_data_pair()
        
        # In a real scenario, we'd need a more robust move encoding,
        # but for this POC, we can treat the move as a discrete label.
        return board_tensor, torch.tensor(move_idx, dtype=torch.long), torch.tensor(conceptual_features, dtype=torch.float32)
      
