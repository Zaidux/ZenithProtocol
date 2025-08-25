# src/models/arlc_controller.py

import torch
import torch.nn as nn
import numpy as np
import hashlib
import random
from typing import Dict, List, Tuple
from ..utils.config import Config
from ..games.tetris_env import get_conceptual_features as get_tetris_features
from ..games.chess_env import get_conceptual_features as get_chess_features
from .hyper_conceptual_thinking import ConceptDiscoveryEngine
from .strategic_planner import StrategicPlanner
from .sswm import SSWM # Import the new SSWM

class ARLCController:
    """
    The Adaptive Reinforcement Learning Controller (ARLC) is the "Brain" of the system.
    It evaluates possible moves and chooses the best one based on a scoring heuristic.
    This version is designed to be domain-agnostic and orchestrates the HCT and EoM processes.
    """
    def __init__(self, strategic_planner: StrategicPlanner, sswm: SSWM, exploration_weight: float = 5.0, eom_weight: float = 2.0):
        self.strategic_planner = strategic_planner
        self.sswm = sswm # New: Pass the SSWM to the ARLC
        self.exploration_weight = exploration_weight
        self.eom_weight = eom_weight
        self.visited_states = {}
        self.reward_coeffs = Config.ARLC_REWARD_COEFFS
        self.cde = ConceptDiscoveryEngine()
        self.last_fused_rep = None
        self.is_exploring = False

    def get_generic_conceptual_features(self, state_shape: tuple) -> np.ndarray:
        """
        Generates a generic, baseline set of conceptual features for a new domain.
        This is the "Initial Analysis" step.
        """
        # For a new, unknown domain, we'll start with a generic set of features
        # such as board dimensions, number of active pieces, etc.
        num_features = 3 # A simple starting point
        conceptual_features = np.zeros(num_features)

        # Simple hypotheses based on board dimensions
        if len(state_shape) == 3: # Assuming a 2D board with channels
            height, width = state_shape[1], state_shape[2]
            conceptual_features[0] = height # Board Height
            conceptual_features[1] = width # Board Width
            # A placeholder for piece count
            conceptual_features[2] = np.sum(state_shape)

        return conceptual_features

    def evaluate_conceptual_features(self, conceptual_features: np.ndarray, fused_representation: torch.Tensor, domain: str) -> Dict[str, float]:
        """
        Calculates a score for a given set of conceptual features and adds an HCT bonus.
        This function is the core of the ARLC's reward mechanism.
        """
        score = 0
        if domain == 'tetris':
            score = (self.reward_coeffs['tetris']['lines_cleared'] * conceptual_features[0]) + \
                    (self.reward_coeffs['tetris']['gaps'] * conceptual_features[1]) + \
                    (self.reward_coeffs['tetris']['max_height'] * conceptual_features[2])
        elif domain == 'chess':
            score = (self.reward_coeffs['chess']['material_advantage'] * conceptual_features[0]) + \
                    (self.reward_coeffs['chess']['king_safety'] * (conceptual_features[1] - conceptual_features[2])) + \
                    (self.reward_coeffs['chess']['center_control'] * (conceptual_features[3] - conceptual_features[4]))
        elif self.is_exploring:
            # During exploration, the conceptual score is initially zero,
            # as we don't know what features are valuable yet.
            score = 0.0

        # State exploration bonus
        state_hash = hashlib.sha256(conceptual_features.tobytes()).hexdigest()
        visits = self.visited_states.get(state_hash, 0)
        exploration_bonus = self.exploration_weight / (1 + visits)
        self.visited_states[state_hash] = visits + 1

        # HCT: Check for newly discovered high-value concepts and add a bonus
        hct_bonus, discovered_concept = self.cde.analyze_for_new_concepts(fused_representation, score, domain)

        # Add the 'surprise bonus' to the conceptual score
        if self.is_exploring and self.last_fused_rep is not None:
            surprise_bonus = self.calculate_eom_bonus(self.last_fused_rep, fused_representation)
            score += surprise_bonus

        final_score = score + exploration_bonus + hct_bonus

        return {
            "conceptual_score": score,
            "exploration_bonus": exploration_bonus,
            "hct_bonus": hct_bonus,
            "discovered_concept": discovered_concept,
            "score": final_score
        }

    def calculate_eom_bonus(self, last_fused_rep: torch.Tensor, current_fused_rep: torch.Tensor) -> float:
        """
        Calculates the Energy of Movement (EoM) bonus.
        This bonus rewards moves that cause a significant conceptual shift.
        """
        conceptual_change = torch.norm(current_fused_rep - last_fused_rep, p=2)
        eom_bonus = self.eom_weight * (conceptual_change.item())
        return eom_bonus

    def choose_move(self, board_state: np.ndarray, domain: str, piece_idx: int | None = None) -> Tuple[int | None, Dict]:
        """
        Chooses a move based on the conceptual evaluation of possible outcomes.
        This function is for live game play, not for the training loop.
        """
        if self.is_exploring:
            # In exploration mode, we choose a random move to gather data
            legal_moves = range(5) # Placeholder for the actual number of legal moves
            chosen_move = random.choice(legal_moves)
            decision_context = {"chosen_move": chosen_move, "chosen_score": 0.0, "all_scores": [0.0]}
            return chosen_move, decision_context

        # This is a placeholder for move generation
        all_scores = [1.5, 2.1, 0.8, 1.9, 2.5] 

        # New: Adjust scores based on the current strategic goal
        adjusted_scores = self.adjust_score_for_strategy(all_scores, board_state, domain)
        
        # New: Further adjust scores based on SSWM's prediction
        adjusted_scores = self.predictive_score_adjustment(adjusted_scores, self.last_fused_rep, domain)

        chosen_move = np.argmax(adjusted_scores)
        chosen_score = adjusted_scores[chosen_move]

        # The decision context will now also include the current strategic goal
        decision_context = {
            'chosen_move': chosen_move,
            'chosen_score': chosen_score,
            'all_scores': all_scores,
            'current_strategy': self.strategic_planner.current_goal
        }

        return chosen_move, decision_context

    def adjust_score_for_strategy(self, scores: list, game_state: np.ndarray, domain: str) -> list:
        """
        Adjusts the raw scores of each move based on how well they align
        with the current high-level strategic goal.
        """
        conceptual_features_for_goal_selection = self.strategic_planner.model.get_conceptual_features(game_state)
        strategic_goal = self.strategic_planner.select_goal(conceptual_features_for_goal_selection, domain)

        # A simple, rule-based bonus system for demonstration.
        # This would be much more complex in a real implementation.
        if strategic_goal['goal'] == 'control_center' and domain == 'chess':
            # For a real implementation, you would need to simulate each move
            # and evaluate its conceptual features.
            # Here, we just give a simple bonus to a hypothetical "good" move.
            bonus_move_index = 2
            if bonus_move_index < len(scores):
                scores[bonus_move_index] += 1.0 # Add a bonus for a strategically good move.

        return scores
        
    def predictive_score_adjustment(self, scores: list, current_fused_rep: torch.Tensor, domain: str) -> list:
        """
        Uses the SSWM to predict future outcomes and adjust move scores.
        """
        for i in range(len(scores)):
            # Simulate the outcome of each move and get a predicted reward
            predicted_rep, predicted_reward = self.sswm.simulate_what_if_scenario(
                start_state_rep=current_fused_rep,
                hypothetical_move=i,
                num_steps=1 # Look one step ahead
            )
            
            # The ARLC can now add a bonus based on the predicted reward
            scores[i] += predicted_reward
            
        return scores
