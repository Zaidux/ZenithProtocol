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
from .sswm import SSWM
# New Imports
from ..conceptual_knowledge_graph.ckg import ConceptualKnowledgeGraph
from ..web_access.web_access import WebAccess

class ARLCController:
    """
    The Adaptive Reinforcement Learning Controller (ARLC) is the "Brain" of the system.
    It evaluates possible moves and chooses the best one based on a scoring heuristic.
    This version is designed to be domain-agnostic and orchestrates the HCT and EoM processes.
    It now integrates with the Conceptual Knowledge Graph and the Web Access module.
    """
    def __init__(self, 
                 strategic_planner: StrategicPlanner, 
                 sswm: SSWM, 
                 exploration_weight: float = 5.0, 
                 eom_weight: float = 2.0,
                 ckg: ConceptualKnowledgeGraph = None, # New: CKG instance
                 web_access: WebAccess = None):     # New: WebAccess instance
        self.strategic_planner = strategic_planner
        self.sswm = sswm
        self.exploration_weight = exploration_weight
        self.eom_weight = eom_weight
        self.visited_states = {}
        self.reward_coeffs = Config.ARLC_REWARD_COEFFS
        self.cde = ConceptDiscoveryEngine()
        self.last_fused_rep = None
        self.is_exploring = False
        
        # New: Initialize the CKG and Web Access modules
        self.ckg = ckg or ConceptualKnowledgeGraph()
        self.web_access = web_access or WebAccess()

    def get_generic_conceptual_features(self, state_shape: tuple) -> np.ndarray:
        """
        Generates a generic, baseline set of conceptual features for a new domain.
        This is the "Initial Analysis" step.
        """
        num_features = 3
        conceptual_features = np.zeros(num_features)

        if len(state_shape) == 3:
            height, width = state_shape[1], state_shape[2]
            conceptual_features[0] = height
            conceptual_features[1] = width
            conceptual_features[2] = np.sum(state_shape)

        return conceptual_features
    
    # New Method
    def check_for_knowledge_gaps(self, prompt: str) -> List[str]:
        """
        Analyzes a prompt to identify concepts that are not in the CKG.
        """
        gaps = []
        # This is a simplified keyword-based check.
        # A more advanced model would use the Conceptual Encoder to find gaps.
        words = set(prompt.lower().split())
        for word in words:
            if not self.ckg.query(word):
                gaps.append(word)
        return gaps

    # New Method
    def update_knowledge_with_web_data(self, query: str):
        """
        Uses the WebAccess module to get real-time data and updates the CKG.
        """
        # Check if the data is stale before making a new web call
        if self.web_access.check_for_update(query, time_limit_minutes=1440): # Update every 24 hours
            print(f"\n[ARLC] Knowledge for '{query}' is stale. Performing web search...")
            summary = self.web_access.search_and_summarize(query)
            if summary:
                # This is a simple update logic. A more complex one would parse the summary
                # into a structured knowledge graph format.
                self.ckg.add_node(query, {"type": "concept", "source": "web_search", "content": summary})
                print(f"[ARLC] CKG updated with new information from the web.")
            else:
                print(f"[ARLC] Web search for '{query}' found no relevant information.")

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
            score = 0.0

        state_hash = hashlib.sha256(conceptual_features.tobytes()).hexdigest()
        visits = self.visited_states.get(state_hash, 0)
        exploration_bonus = self.exploration_weight / (1 + visits)
        self.visited_states[state_hash] = visits + 1

        hct_bonus, discovered_concept = self.cde.analyze_for_new_concepts(fused_representation, score, domain)

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
        """
        # New: Before choosing a move, check for knowledge gaps and update CKG
        # This is a simplified example; a real-world scenario would use a text prompt
        # but for game environments, we can infer a "knowledge gap" from a low-confidence
        # prediction or an unknown state.
        if random.random() < 0.1: # Simulate a 10% chance of needing a web search
            query = "latest AI news" # Example query
            self.update_knowledge_with_web_data(query)

        if self.is_exploring:
            legal_moves = range(5)
            chosen_move = random.choice(legal_moves)
            decision_context = {"chosen_move": chosen_move, "chosen_score": 0.0, "all_scores": [0.0]}
            return chosen_move, decision_context

        all_scores = [1.5, 2.1, 0.8, 1.9, 2.5] 

        adjusted_scores = self.adjust_score_for_strategy(all_scores, board_state, domain)

        adjusted_scores = self.predictive_score_adjustment(adjusted_scores, self.last_fused_rep, domain)

        chosen_move = np.argmax(adjusted_scores)
        chosen_score = adjusted_scores[chosen_move]

        decision_context = {
            'chosen_move': chosen_move,
            'chosen_score': chosen_score,
            'all_scores': all_scores,
            'current_strategy': self.strategic_planner.current_goal
        }

        # New: Store the decision in the CKG for long-term memory
        self.ckg.add_prompt_response(
            prompt=f"Board State Hash: {hashlib.sha256(board_state.tobytes()).hexdigest()}",
            response=f"Chosen Move: {chosen_move} with score {chosen_score}",
            concepts=[domain, "move", str(chosen_move), "score"]
        )

        return chosen_move, decision_context

    def adjust_score_for_strategy(self, scores: list, game_state: np.ndarray, domain: str) -> list:
        """
        Adjusts the raw scores of each move based on how well they align
        with the current high-level strategic goal.
        """
        conceptual_features_for_goal_selection = self.strategic_planner.model.get_conceptual_features(game_state)
        strategic_goal = self.strategic_planner.select_goal(conceptual_features_for_goal_selection, domain)

        if strategic_goal['goal'] == 'control_center' and domain == 'chess':
            bonus_move_index = 2
            if bonus_move_index < len(scores):
                scores[bonus_move_index] += 1.0

        return scores

    def predictive_score_adjustment(self, scores: list, current_fused_rep: torch.Tensor, domain: str) -> list:
        """
        Uses the SSWM to predict future outcomes and adjust move scores.
        """
        for i in range(len(scores)):
            predicted_rep, predicted_reward = self.sswm.simulate_what_if_scenario(
                start_state_rep=current_fused_rep,
                hypothetical_move=i,
                num_steps=1
            )
            scores[i] += predicted_reward

        return scores

    def rapid_adaptation_to_new_domain(self, new_domain_data: List[Dict]):
        """
        This is a placeholder for the rapid adaptation process.
        It simulates a few fast updates on the new domain data.
        """
        print("Rapidly adapting to the new domain with meta-learned knowledge...")
        optimizer = torch.optim.Adam(self.strategic_planner.model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()

        for i, data_point in enumerate(new_domain_data):
            state = data_point['state']
            conceptual_features = data_point['conceptual_features']
            target = data_point['target']

            state_tensor = torch.tensor(state).unsqueeze(0).float().to(Config.DEVICE)
            conceptual_tensor = torch.tensor(conceptual_features).unsqueeze(0).float().to(Config.DEVICE)
            target_tensor = torch.tensor(target).unsqueeze(0).float().to(Config.DEVICE)

            predicted_output, _, _ = self.strategic_planner.model(state_tensor, conceptual_tensor, data_point['domain'])
            loss = criterion(predicted_output, target_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"  - Adaptation step {i+1} complete. Loss: {loss.item():.4f}")

        print("Rapid adaptation complete.")
        self.is_exploring = True
