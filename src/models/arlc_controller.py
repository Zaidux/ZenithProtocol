# /src/models/arlc_controller.py

import torch
import torch.nn as nn
import numpy as np
import hashlib
from typing import Dict, List, Tuple
from ..games.tetris_env import BOARD_WIDTH, SHAPES, place_piece, evaluate_conceptual_features

class ARLCController:
    """
    The Adaptive Reinforcement Learning Controller (ARLC) is the "Brain" of the system.
    It evaluates possible moves and chooses the best one based on a scoring heuristic.
    """
    def __init__(self,
                 points_per_line: float = 10.0,
                 gap_penalty: float = 2.0,
                 height_penalty: float = 1.0,
                 exploration_weight: float = 5.0):
        
        self.points_per_line = points_per_line
        self.gap_penalty = gap_penalty
        self.height_penalty = height_penalty
        self.exploration_weight = exploration_weight
        self.visited_states = {} # To track visited board states for exploration bonus

    def evaluate_board_state(self, board: np.ndarray) -> Dict[str, float]:
        """
        Calculates a score for a given board state based on a heuristic.
        This is a deterministic reward function for the RL agent.
        """
        lines_cleared = np.sum(np.all(board != 0, axis=1))
        gaps = self._count_gaps(board)
        max_height = self._get_max_height(board)

        base_score = (self.points_per_line * lines_cleared) - \
                     (self.gap_penalty * gaps) - \
                     (self.height_penalty * max_height)
        
        # Exploration bonus based on state visitation
        state_hash = hashlib.sha256(board.tobytes()).hexdigest()
        visits = self.visited_states.get(state_hash, 0)
        exploration_bonus = self.exploration_weight / (1 + visits)
        self.visited_states[state_hash] = visits + 1
        
        final_score = base_score + exploration_bonus

        return {
            "lines_cleared": float(lines_cleared),
            "gaps": float(gaps),
            "max_height": float(max_height),
            "base_score": base_score,
            "exploration_bonus": exploration_bonus,
            "score": final_score
        }

    def _count_gaps(self, board: np.ndarray) -> int:
        gaps = 0
        for col in range(board.shape[1]):
            filled_found = False
            for row in range(board.shape[0]):
                if board[row, col] != 0:
                    filled_found = True
                elif filled_found:
                    gaps += 1
        return gaps

    def _get_max_height(self, board: np.ndarray) -> int:
        max_height = 0
        for col in range(board.shape[1]):
            for row in range(board.shape[0]):
                if board[row, col] != 0:
                    col_height = board.shape[0] - row
                    max_height = max(max_height, col_height)
                    break
        return max_height

    def choose_move(self, current_board: np.ndarray, piece_idx: int, temperature: float = 1.0) -> Tuple[int, np.ndarray, Dict]:
        """
        Evaluates all possible moves for the current piece and chooses one using a softmax distribution.
        This allows for a balance between exploration and exploitation.
        """
        piece_to_place = SHAPES[piece_idx]
        possible_moves = []
        for x_pos in range(BOARD_WIDTH - len(piece_to_place[0]) + 1):
            board_after = place_piece(current_board, piece_to_place, x_pos)
            if board_after is not None:
                possible_moves.append((x_pos, board_after))

        if not possible_moves:
            return -1, current_board, {"error": "No valid moves found."}

        scores = [self.evaluate_board_state(m[1])['score'] for m in possible_moves]
        
        # Softmax to choose a move probabilistically
        exp_scores = np.exp(np.array(scores) / temperature)
        probabilities = exp_scores / np.sum(exp_scores)
        chosen_index = np.random.choice(len(possible_moves), p=probabilities)
        chosen_move, chosen_board_state = possible_moves[chosen_index]
        
        decision_context = {
            'chosen_move': chosen_move,
            'chosen_score': scores[chosen_index],
            'all_scores': scores,
            'all_probabilities': probabilities,
            'all_moves': [m[0] for m in possible_moves],
            'move_metrics': [self.evaluate_board_state(m[1]) for m in possible_moves]
        }
        
        return chosen_move, chosen_board_state, decision_context
