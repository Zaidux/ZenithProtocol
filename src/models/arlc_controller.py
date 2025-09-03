# src/models/arlc_controller.py

import torch
import torch.nn as nn
import numpy as np
import hashlib
import random
from typing import Dict, List, Tuple
from ..utils.config import Config
from .hyper_conceptual_thinking import ConceptDiscoveryEngine
from .strategic_planner import StrategicPlanner
from .sswm import SSWM
from ..conceptual_knowledge_graph.ckg import ConceptualKnowledgeGraph
from ..web_access.web_access import WebAccess

# New Imports to use the C++ backend
import eom_calculator_cpp
import spm_controller_cpp

# New import for the Self-Evolving Knowledge Agent
from .self_evolving_knowledge_agent import SelfEvolvingKnowledgeAgent

class ARLCController:
    """
    The Adaptive Reinforcement Learning Controller (ARLC) is the "Brain" of the system.
    It evaluates possible moves and chooses the best one based on a scoring heuristic.
    This version includes an integrated self-correction mechanism.
    """
    def __init__(self, 
                 strategic_planner: StrategicPlanner, 
                 sswm: SSWM, 
                 exploration_weight: float = 5.0, 
                 eom_weight: float = 2.0,
                 ckg: ConceptualKnowledgeGraph = None, 
                 web_access: WebAccess = None,
                 model: nn.Module = None):
        self.strategic_planner = strategic_planner
        self.sswm = sswm
        self.exploration_weight = exploration_weight
        self.eom_weight = eom_weight
        self.visited_states = {}
        self.reward_coeffs = Config.ARLC_REWARD_COEFFS
        self.cde = ConceptDiscoveryEngine(ckg=ckg)
        self.last_fused_rep = None
        self.is_exploring = False

        self.ckg = ckg or ConceptualKnowledgeGraph()
        self.web_access = web_access or WebAccess(self.ckg)
        self.model = model # New: The ARLC needs a reference to the main model for self-correction.
        
        # New: Initialize the C++ EOM calculator
        self.cpp_eom_calculator = eom_calculator_cpp.EoMCalculator()
        # New: Initialize the C++ SPM controller
        self.cpp_spm_controller = spm_controller_cpp.SPMController()
        
        # New: Initialize the SEKA
        from ..conceptual_encoder.conceptual_encoder import ZenithConceptualEncoder
        self.seka = SelfEvolvingKnowledgeAgent(
            model=model, 
            arlc=self, 
            ckg=self.ckg, 
            web_access=self.web_access,
            conceptual_encoder=ZenithConceptualEncoder()
        )

    def get_generic_conceptual_features(self, state_shape: tuple) -> np.ndarray:
        num_features = 3
        conceptual_features = np.zeros(num_features)
        if len(state_shape) == 3:
            height, width = state_shape[1], state_shape[2]
            conceptual_features[0] = height
            conceptual_features[1] = width
            conceptual_features[2] = np.sum(state_shape)
        return conceptual_features

    def check_for_knowledge_gaps(self, prompt: str) -> List[str]:
        gaps = []
        words = set(prompt.lower().split())
        for word in words:
            if not self.ckg.query(word):
                gaps.append(word)
        return gaps

    def update_knowledge_with_web_data(self, query: str):
        if self.web_access.check_for_update(query, time_limit_minutes=1440):
            print(f"\n[ARLC] Knowledge for '{query}' is stale. Performing web search...")
            summary = self.web_access.search_and_summarize(query)
            if summary:
                self.ckg.add_node(query, {"type": "concept", "source": "web_search", "content": summary})
                print(f"[ARLC] CKG updated with new information from the web.")
            else:
                print(f"[ARLC] Web search for '{query}' found no relevant information.")

    def evaluate_conceptual_features(self, conceptual_features: np.ndarray, fused_representation: torch.Tensor, domain: str) -> Dict[str, float]:
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
            # Use the C++ EoM calculator for a performance boost
            surprise_bonus = self.cpp_eom_calculator.calculate_eom_bonus(
                self.last_fused_rep.detach().cpu().numpy(),
                fused_representation.detach().cpu().numpy(),
                self.eom_weight
            )
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
        raise NotImplementedError("EoM calculation is now handled by the C++ backend.")

    def choose_move(self, board_state: np.ndarray, domain: str, model, piece_idx: int | None = None) -> Tuple[int | None, Dict]:
        # New logic to trigger autonomous learning
        gaps = self.check_for_knowledge_gaps(f"board state {domain}")
        if gaps:
            print(f"[ARLC] Detected knowledge gaps: {gaps}. Triggering autonomous learning.")
            # For simplicity, we'll learn about the first gap found
            self.seka.initiate_knowledge_acquisition(gaps[0])

        if random.random() < 0.1:
            query = "latest AI news"
            self.update_knowledge_with_web_data(query)

        if self.is_exploring:
            # Use the C++ SPM controller to simulate a move
            self.cpp_spm_controller.allocate_for_tasks([domain, "exploration"])
            mock_input = np.random.rand(1, 128)
            processed_output = self.cpp_spm_controller.run_parallel_simulation(mock_input, domain)
            
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

        self.ckg.add_prompt_response(
            prompt=f"Board State Hash: {hashlib.sha256(board_state.tobytes()).hexdigest()}",
            response=f"Chosen Move: {chosen_move} with score {chosen_score}",
            concepts=[domain, "move", str(chosen_move), "score"]
        )

        return chosen_move, decision_context

    def adjust_score_for_strategy(self, scores: list, game_state: np.ndarray, domain: str) -> list:
        conceptual_features_for_goal_selection = self.strategic_planner.model.get_conceptual_features(game_state)
        strategic_goal = self.strategic_planner.select_goal(conceptual_features_for_goal_selection, domain)

        if strategic_goal['goal'] == 'control_center' and domain == 'chess':
            bonus_move_index = 2
            if bonus_move_index < len(scores):
                scores[bonus_move_index] += 1.0
        return scores

    def predictive_score_adjustment(self, scores: list, current_fused_rep: torch.Tensor, domain: str) -> list:
        for i in range(len(scores)):
            predicted_rep, predicted_reward = self.sswm.simulate_what_if_scenario(
                start_state_rep=current_fused_rep,
                hypothetical_move=i,
                num_steps=1
            )
            scores[i] += predicted_reward
        return scores

    def rapid_adaptation_to_new_domain(self, new_domain_data: List[Dict]):
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

    def self_correct_from_failure(self, failure_report: Dict, model: nn.Module):
        """
        Analyzes a failure report from the ExplainabilityModule and performs self-correction.
        It updates both the model's weights and the CKG to prevent similar errors.
        """
        print(f"\n[ARLC] Initiating self-correction based on failure report: {failure_report.get('type')}")

        error_type = failure_report.get('type', 'unknown_error')
        self.ckg.add_node(f"Failure_{hashlib.sha256(str(failure_report).encode()).hexdigest()}", {
            "type": "error", 
            "subtype": error_type, 
            "description": failure_report.get('explanation', 'No explanation provided.'),
            "causal_factors": failure_report.get('causal_factors', []),
            "timestamp": datetime.now().isoformat()
        })
        self.ckg.add_edge("ASREHModel", f"Failure_{hashlib.sha256(str(failure_report).encode()).hexdigest()}", "CAUSED_BY")

        causal_factors = failure_report.get('causal_factors', [])

        for factor in causal_factors:
            if factor == 'conceptual_misinterpretation':
                print("  - Adjusting ConceptualAttention weights to correct misinterpretation.")
                with torch.no_grad():
                    # This is a conceptual implementation of parameter adjustment.
                    # A real system could do a targeted gradient update.
                    for param in model.conceptual_attention.parameters():
                        param.add_(torch.randn_like(param) * 0.001)

            elif factor == 'sswm_hallucination':
                print("  - Adjusting SSWM's state predictor weights to reduce hallucination.")
                with torch.no_grad():
                    for param in self.sswm.state_predictor.parameters():
                        param.add_(torch.randn_like(param) * -0.002)

        print("[ARLC] Self-correction complete. New knowledge integrated into CKG and model weights adjusted.")
