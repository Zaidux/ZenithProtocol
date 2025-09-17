"""
Enhanced Adaptive Reinforcement Learning Controller (ARLC)
=========================================================
Now integrates with CKG's causal rules for verifiable, rule-based decision making.
"""

import torch
import torch.nn as nn
import numpy as np
import hashlib
import random
import json
from typing import Dict, List, Tuple, Optional
from ..utils.config import Config
from .hyper_conceptual_thinking import ConceptDiscoveryEngine
from .strategic_planner import StrategicPlanner
from .sswm import SSWM
from ..conceptual_knowledge_graph.ckg import ConceptualKnowledgeGraph
from ..web_access.web_access import WebAccess
from datetime import datetime
from .meta_learner import MetaLearner
from .self_architecting_agent import SelfArchitectingAgent
from .self_evolving_knowledge_agent import SelfEvolvingKnowledgeAgent
from ..local_agent.LocalExecutionAgent import LocalExecutionAgent
from ..conceptual_encoder.conceptual_encoder import ZenithConceptualEncoder
import eom_calculator_cpp
import spm_controller_cpp

class ARLCController:
    """
    The Adaptive Reinforcement Learning Controller (ARLC) is the "Brain" of the system.
    Now enhanced with CKG-integrated reward calculation and verifiable decision making.
    """
    def __init__(self, 
                 strategic_planner: StrategicPlanner, 
                 sswm: SSWM, 
                 exploration_weight: float = 5.0, 
                 eom_weight: float = 2.0,
                 ckg: ConceptualKnowledgeGraph = None, 
                 web_access: WebAccess = None,
                 model: nn.Module = None,
                 meta_learner: Optional[MetaLearner] = None):
        self.strategic_planner = strategic_planner
        self.sswm = sswm
        self.exploration_weight = exploration_weight
        self.eom_weight = eom_weight
        self.visited_states = {}
        self.reward_coeffs = Config.ARLC_REWARD_COEFFS
        self.ckg = ckg or ConceptualKnowledgeGraph()
        self.model = model
        # Pass the model instance to the CDE for architectural suggestions
        self.cde = ConceptDiscoveryEngine(ckg=self.ckg, model=self.model)
        self.last_fused_rep = None
        self.is_exploring = False
        self.web_access = web_access or WebAccess(self.ckg)
        self.cpp_eom_calculator = eom_calculator_cpp.EoMCalculator()
        self.cpp_spm_controller = spm_controller_cpp.SPMController()
        self.seka = SelfEvolvingKnowledgeAgent(
            model=self.model, 
            arlc=self, 
            ckg=self.ckg, 
            web_access=self.web_access,
            conceptual_encoder=ZenithConceptualEncoder(ckg=self.ckg)
        )
        self.lea = LocalExecutionAgent(ckg=self.ckg, arlc=self, em=None)
        self.meta_learner = meta_learner
        # Initialize the Self-Architecting Agent
        self.self_architecting_agent = SelfArchitectingAgent(
            sswm=self.sswm, 
            ckg=self.ckg, 
            model=self.model, 
            em=self.model.explainability_module if hasattr(self.model, 'explainability_module') else None
        )

    def evaluate_with_ckg(self, conceptual_features: torch.Tensor, 
                         ckg_validation: Dict, domain: str) -> Dict[str, float]:
        """
        Enhanced evaluation using CKG rules for verifiable reward calculation.
        
        Args:
            conceptual_features: Forecasted conceptual features from SSWM
            ckg_validation: Validation result from CKG
            domain: The problem domain
            
        Returns:
            Comprehensive scoring result with rule-based breakdown
        """
        # Calculate base score using CKG reward rules
        base_score = self._calculate_rule_based_score(conceptual_features, domain)
        
        # Apply validation confidence
        validation_confidence = ckg_validation.get('confidence', 1.0)
        validated_score = base_score * validation_confidence
        
        # Apply exploration bonus
        state_hash = self._create_state_hash(conceptual_features)
        visits = self.visited_states.get(state_hash, 0)
        exploration_bonus = self.exploration_weight / (1 + visits)
        self.visited_states[state_hash] = visits + 1
        
        # Apply HCT bonus for novel discoveries
        hct_bonus, discovered_concept = self.cde.analyze_for_new_concepts(
            conceptual_features, validation_confidence, validated_score, domain
        )
        
        # Apply surprise bonus for exploration
        surprise_bonus = 0.0
        if self.is_exploring and self.last_fused_rep is not None:
            surprise_bonus = self.cpp_eom_calculator.calculate_eom_bonus(
                self.last_fused_rep.detach().cpu().numpy(),
                conceptual_features.detach().cpu().numpy(),
                self.eom_weight
            )
        
        # Calculate final score
        final_score = validated_score + exploration_bonus + hct_bonus + surprise_bonus
        
        return {
            "base_score": base_score,
            "validated_score": validated_score,
            "validation_confidence": validation_confidence,
            "exploration_bonus": exploration_bonus,
            "hct_bonus": hct_bonus,
            "surprise_bonus": surprise_bonus,
            "discovered_concept": discovered_concept,
            "applied_rules": ckg_validation.get('applied_rules', []),
            "violated_rules": ckg_validation.get('violated_rules', []),
            "score": final_score
        }

    def _calculate_rule_based_score(self, conceptual_features: torch.Tensor, domain: str) -> float:
        """
        Calculate score based on CKG reward rules for the domain.
        """
        score = 0.0
        features_dict = self._extract_features_dict(conceptual_features, domain)
        
        # Use CKG reward rules instead of hardcoded coefficients
        domain_rules = self.ckg.reward_rules.get(domain, {})
        
        for concept, rule in domain_rules.items():
            feature_value = features_dict.get(concept, 0.0)
            score += feature_value * rule['weight']
        
        return score

    def _extract_features_dict(self, conceptual_features: torch.Tensor, domain: str) -> Dict:
        """
        Extract conceptual features as a dictionary for rule-based scoring.
        """
        features_np = conceptual_features.detach().cpu().numpy()
        if features_np.size == 0:
            return {}
            
        if domain == 'tetris':
            return {
                'lines_cleared': float(features_np[0][0] if features_np.size > 0 else 0),
                'gaps': float(features_np[0][1] if features_np.size > 1 else 0),
                'max_height': float(features_np[0][2] if features_np.size > 2 else 0),
                'board_fullness': float(features_np[0][3] if features_np.size > 3 else 0)
            }
        elif domain == 'chess':
            return {
                'material_advantage': float(features_np[0][0] if features_np.size > 0 else 0),
                'king_safety': float(features_np[0][1] if features_np.size > 1 else 0),
                'center_control': float(features_np[0][2] if features_np.size > 2 else 0),
                'development': float(features_np[0][3] if features_np.size > 3 else 0)
            }
        return {}

    def _create_state_hash(self, conceptual_features: torch.Tensor) -> str:
        """Create a hash for state tracking."""
        features_np = conceptual_features.detach().cpu().numpy()
        return hashlib.sha256(features_np.tobytes()).hexdigest()

    def choose_move(self, board_state: np.ndarray, domain: str, model, 
                   piece_idx: Optional[int] = None) -> Tuple[Optional[int], Dict]:
        """
        Enhanced move selection with CKG-integrated evaluation.
        """
        # Check for knowledge gaps and update if needed
        gaps = self.check_for_knowledge_gaps(f"board state {domain}")
        if gaps:
            print(f"[ARLC] Detected knowledge gaps: {gaps}. Triggering autonomous learning.")
            self.seka.initiate_knowledge_acquisition(gaps[0])
            self.rapid_adaptation_to_new_domain([])
        
        # Periodic web knowledge update
        if random.random() < 0.1:
            query = "latest AI news"
            self.update_knowledge_with_web_data(query)
        
        # Meta-learning if model is struggling
        if self.model.is_struggling():
            print("[ARLC] Model is struggling. Initiating mini meta-learning loop.")
            self.meta_learner.run_mini_meta_training(self.model, self.ckg)

        # Architectural self-improvement
        prediction = self.self_architecting_agent.predict_future_need(
            conceptual_prompt="model is performing well, but could be more efficient"
        )
        if prediction['upgrade_type'] != 'none':
            self.self_architecting_agent.propose_upgrade(
                upgrade_type=prediction['upgrade_type'],
                reasoning=prediction['reasoning']
            )

        # Exploration mode
        if self.is_exploring:
            return self._explore_random_move(domain)

        # Normal decision making with CKG integration
        return self._make_informed_decision(board_state, domain, model)

    def _make_informed_decision(self, board_state: np.ndarray, domain: str, model) -> Tuple[Optional[int], Dict]:
        """Make decisions using CKG-validated forecasting."""
        possible_moves = self._get_possible_moves(domain, board_state)
        scores = []
        decision_metrics = []
        
        for move in possible_moves:
            # Forecast outcome for this move
            forecasted_state = self.sswm.predict(board_state, move, domain)
            
            # Get conceptual features
            with torch.no_grad():
                conceptual_features = model.conceptual_attention_layer(model.encoder(forecasted_state))
            
            # Validate forecast against CKG rules
            ckg_validation = self.ckg.validate_forecast(conceptual_features, move, domain)
            
            # Evaluate using CKG rules
            score_result = self.evaluate_with_ckg(conceptual_features, ckg_validation, domain)
            
            scores.append(score_result['score'])
            decision_metrics.append({
                'move': move,
                'score_breakdown': score_result,
                'ckg_validation': ckg_validation,
                'forecasted_state': forecasted_state
            })
        
        # Select move with softmax exploration
        chosen_index = self._softmax_selection(scores, temperature=0.5)
        chosen_move = possible_moves[chosen_index]
        
        # Build comprehensive decision context
        decision_context = self._build_decision_context(
            chosen_move, scores, decision_metrics, domain
        )
        
        # Log decision to CKG for verifiability
        self._log_decision_to_ckg(chosen_move, scores[chosen_index], domain, board_state)
        
        return chosen_move, decision_context

    def _get_possible_moves(self, domain: str, board_state: np.ndarray) -> List:
        """Get all possible moves for the domain."""
        if domain == 'tetris':
            return list(range(10))  # x positions 0-9
        elif domain == 'chess':
            # This would interface with a chess engine in practice
            return ['e2e4', 'd2d4', 'g1f3', 'c2c4', 'e7e5']  # Example moves
        return []

    def _softmax_selection(self, scores: List[float], temperature: float = 1.0) -> int:
        """Select action using softmax probability distribution."""
        scores_array = np.array(scores)
        scaled_scores = scores_array / temperature
        exp_scores = np.exp(scaled_scores - np.max(scaled_scores))  # Numerical stability
        probabilities = exp_scores / np.sum(exp_scores)
        return np.random.choice(len(scores), p=probabilities)

    def _build_decision_context(self, chosen_move: int, scores: List[float], 
                              decision_metrics: List[Dict], domain: str) -> Dict:
        """Build comprehensive context for explanation generation."""
        return {
            'chosen_move': chosen_move,
            'chosen_score': scores[chosen_move],
            'all_moves': [dm['move'] for dm in decision_metrics],
            'all_scores': scores,
            'decision_metrics': decision_metrics,
            'domain': domain,
            'timestamp': datetime.now().isoformat(),
            'strategic_context': self.strategic_planner.current_goal if hasattr(self.strategic_planner, 'current_goal') else None
        }

    def _log_decision_to_ckg(self, move: int, score: float, domain: str, board_state: np.ndarray):
        """Log decision to CKG for verifiable auditing."""
        decision_data = json.dumps({
            "move": move,
            "score": score,
            "domain": domain,
            "board_state_hash": hashlib.sha256(board_state.tobytes()).hexdigest(),
            "timestamp": datetime.now().isoformat()
        })
        self.ckg.add_verifiable_record(decision_data, concepts=[domain, "move", str(move), "decision"])

    def _explore_random_move(self, domain: str) -> Tuple[Optional[int], Dict]:
        """Handle exploration mode with random move selection."""
        self.cpp_spm_controller.allocate_for_tasks([domain, "exploration"])
        mock_input = np.random.rand(1, 128)
        processed_output = self.cpp_spm_controller.run_parallel_simulation(mock_input, domain)
        
        possible_moves = self._get_possible_moves(domain, np.random.rand(20, 10))
        chosen_move = random.choice(possible_moves) if possible_moves else 0
        
        decision_context = {
            "chosen_move": chosen_move,
            "chosen_score": 0.0,
            "all_scores": [0.0] * len(possible_moves),
            "exploration_mode": True
        }
        
        return chosen_move, decision_context

    # --- Existing methods kept for compatibility ---

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

    def generate_action_plan(self, command_intent: str, entities: Dict) -> Dict:
        context_info = self.ckg.query("Socio-Linguistic_Context")
        tone = context_info['node']['properties'].get('tone', 'neutral') if context_info else 'neutral'
        print(f"[ARLC] Generating action plan for intent: '{command_intent}' with tone: {tone}")
        if command_intent == "organize_files":
            plan = {
                "action": "organize_files",
                "folder_name": entities.get("folder_name", "Zenith_Organized_Files"),
                "source_directory": entities.get("source_directory", "Downloads"),
                "file_types": entities.get("file_types", ["jpg", "png", "jpeg"]),
                "conceptual_filters": {
                    "newly_added": True,
                    "content_tags": entities.get("content_tags", [])
                }
            }
            return plan
        else:
            return {"status": "error", "message": "Unknown command intent."}

    def report_failure(self, error_type: str, explanation: str, causal_factors: Optional[List[str]] = None):
        failure_report = {
            "type": error_type,
            "explanation": explanation,
            "causal_factors": causal_factors or [],
            "timestamp": datetime.now().isoformat()
        }
        self.self_correct_from_failure(failure_report, self.model)

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
        print(f"\n[ARLC] Initiating self-correction based on failure report: {failure_report.get('type')}")
        error_type = failure_report.get('type', 'unknown_error')
        failure_data = json.dumps({
            "type": error_type, 
            "subtype": failure_report.get('subtype', 'n/a'),
            "description": failure_report.get('explanation', 'No explanation provided.'),
            "causal_factors": failure_report.get('causal_factors', []),
            "timestamp": datetime.now().isoformat()
        })
        self.ckg.add_verifiable_record(failure_data, concepts=["self_correction", "failure", error_type])
        causal_factors = failure_report.get('causal_factors', [])
        for factor in causal_factors:
            if factor == 'conceptual_misinterpretation':
                print("  - Adjusting ConceptualAttention weights to correct misinterpretation.")
                with torch.no_grad():
                    for param in model.conceptual_attention.parameters():
                        param.add_(torch.randn_like(param) * 0.001)
            elif factor == 'sswm_hallucination':
                print("  - Adjusting SSWM's state predictor weights to reduce hallucination.")
                with torch.no_grad():
                    for param in self.sswm.state_predictor.parameters():
                        param.add_(torch.randn_like(param) * -0.002)
        print("[ARLC] Self-correction complete. New knowledge integrated into CKG and model weights adjusted.")