"""
ASREH Algorithm Orchestrator
=============================
The core implementation of the Adaptive Self-Regulating Explainable Hybrid algorithm.
This module coordinates all components to transform input states into actions with explanations.
"""

import torch
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
from ..conceptual_knowledge_graph.ckg import ConceptualKnowledgeGraph
from ..models.arlc_controller import ARLCController
from ..models.explainability_module import ExplainabilityModule
from ..models.sswm import SSWM
from ..models.asreh_model import ASREHModel
from ..web_access.web_access import WebAccess

class ASREHAlgorithm:
    """
    The central orchestrator that implements the 4-step ASREH algorithm:
    1. Perception & Conceptual Encoding
    2. Simulation & Forecasting  
    3. Evaluation & Decision-Making
    4. Explanation Generation
    """
    
    def __init__(self, model: ASREHModel, sswm: SSWM, arlc: ARLCController, 
                 em: ExplainabilityModule, ckg: ConceptualKnowledgeGraph, 
                 web_access: WebAccess):
        """
        Initialize the ASREH algorithm with all required components.
        
        Args:
            model: The ASREH predictive model for encoding and conceptual attention
            sswm: Self-Supervised World Model for state prediction
            arlc: Adaptive Reinforcement Learning Controller for decision making
            em: Explainability Module for generating human-readable explanations
            ckg: Conceptual Knowledge Graph for causal reasoning and validation
            web_access: Web access component for real-time knowledge updates
        """
        self.model = model
        self.sswm = sswm
        self.arlc = arlc
        self.em = em
        self.ckg = ckg
        self.web_access = web_access
        self.domain_specific_config = {
            'tetris': {'action_range': range(10)},  # x positions 0-9
            'chess': {'action_range': []}  # Will be populated dynamically
        }

    def execute_algorithm_step(self, state: torch.Tensor, domain: str, 
                              current_context: Optional[Dict] = None) -> Tuple[Any, str, Dict]:
        """
        Execute one complete step of the ASREH algorithm.
        
        Args:
            state: Current state representation (e.g., board image tensor)
            domain: The problem domain ('tetris', 'chess', etc.)
            current_context: Optional context from previous steps
            
        Returns:
            Tuple of (chosen_action, explanation, full_decision_context)
        """
        # Step 1: Perception & Conceptual Encoding
        print(f"[ASREH] Step 1: Encoding state and extracting conceptual features...")
        latent_representation, conceptual_features = self._perception_and_encoding(state, domain)
        
        # Step 2: Simulation & Forecasting
        print(f"[ASREH] Step 2: Simulating all possible actions...")
        forecasts = self._simulation_and_forecasting(state, domain, latent_representation)
        
        # Step 3: Evaluation & Decision-Making
        print(f"[ASREH] Step 3: Evaluating actions and making decision...")
        chosen_action, scores, decision_metrics = self._evaluation_and_decision(
            forecasts, conceptual_features, domain
        )
        
        # Step 4: Explanation Generation
        print(f"[ASREH] Step 4: Generating human-readable explanation...")
        decision_context = self._build_decision_context(
            chosen_action, scores, forecasts, decision_metrics, domain, current_context
        )
        explanation = self.em.generate_explanation(decision_context)
        
        return chosen_action, explanation, decision_context

    def _perception_and_encoding(self, state: torch.Tensor, domain: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Step 1: Encode input state and extract conceptual features."""
        # Encode to latent representation
        with torch.no_grad():
            latent_representation = self.model.encoder(state)
            
            # Process through conceptual attention layer
            conceptual_features = self.model.conceptual_attention_layer(latent_representation)
            
        return latent_representation, conceptual_features

    def _simulation_and_forecasting(self, state: torch.Tensor, domain: str, 
                                  latent_rep: torch.Tensor) -> Dict[Any, Dict]:
        """Step 2: Simulate all possible actions and forecast outcomes."""
        forecasts = {}
        possible_actions = self._get_possible_actions(domain, state)
        
        for action in possible_actions:
            # Forecast future state using SSWM
            forecasted_state = self.sswm.predict(state, action, domain)
            
            # Encode forecasted state to get conceptual features
            with torch.no_grad():
                forecasted_latent = self.model.encoder(forecasted_state)
                forecasted_conceptual = self.model.conceptual_attention_layer(forecasted_latent)
            
            # Validate forecast against CKG rules
            ckg_validation = self.ckg.validate_forecast(forecasted_conceptual, action, domain)
            
            forecasts[action] = {
                'state': forecasted_state,
                'conceptual_features': forecasted_conceptual,
                'ckg_validation': ckg_validation,
                'is_valid': ckg_validation.get('is_valid', True)
            }
            
        return forecasts

    def _evaluation_and_decision(self, forecasts: Dict, conceptual_features: torch.Tensor, 
                               domain: str) -> Tuple[Any, Dict, Dict]:
        """Step 3: Evaluate all forecasts and select the best action."""
        scores = {}
        decision_metrics = {}
        
        for action, forecast_data in forecasts.items():
            if not forecast_data['is_valid']:
                scores[action] = -float('inf')  # Invalid actions get lowest score
                continue
                
            # Use ARLC to score based on CKG-validated conceptual features
            score_result = self.arlc.evaluate_with_ckg(
                forecast_data['conceptual_features'], 
                forecast_data['ckg_validation'],
                domain
            )
            
            scores[action] = score_result['final_score']
            decision_metrics[action] = {
                'base_score': score_result['base_score'],
                'exploration_bonus': score_result['exploration_bonus'],
                'hct_bonus': score_result.get('hct_bonus', 0),
                'ckg_validation': forecast_data['ckg_validation']
            }
        
        # Apply softmax selection with temperature for exploration
        chosen_action = self._softmax_action_selection(scores, temperature=0.5)
        
        return chosen_action, scores, decision_metrics

    def _build_decision_context(self, chosen_action: Any, scores: Dict, forecasts: Dict, 
                              decision_metrics: Dict, domain: str, 
                              current_context: Optional[Dict]) -> Dict:
        """Build comprehensive context for explanation generation."""
        return {
            'chosen_action': chosen_action,
            'chosen_score': scores[chosen_action],
            'all_actions': list(scores.keys()),
            'all_scores': [scores[action] for action in scores.keys()],
            'forecasts': forecasts,
            'decision_metrics': decision_metrics,
            'domain': domain,
            'previous_context': current_context,
            'ckg_references': self._extract_ckg_references(forecasts),
            'timestamp': np.datetime64('now')
        }

    def _get_possible_actions(self, domain: str, state: torch.Tensor) -> List:
        """Generate all possible actions for the given domain and state."""
        if domain == 'tetris':
            # For Tetris, actions are x positions (0-9 for a 10-width board)
            return list(range(10))
        elif domain == 'chess':
            # For chess, this would generate legal moves (simplified here)
            # In practice, this would interface with a chess engine
            return ['e2e4', 'd2d4', 'g1f3']  # Example moves
        else:
            raise ValueError(f"Unknown domain: {domain}")

    def _softmax_action_selection(self, scores: Dict, temperature: float = 1.0) -> Any:
        """Select action using softmax probability distribution."""
        actions = list(scores.keys())
        score_values = np.array([scores[action] for action in actions])
        
        # Apply temperature scaling
        scaled_scores = score_values / temperature
        
        # Softmax probabilities
        exp_scores = np.exp(scaled_scores - np.max(scaled_scores))  # Numerical stability
        probabilities = exp_scores / np.sum(exp_scores)
        
        # Sample from distribution
        chosen_index = np.random.choice(len(actions), p=probabilities)
        return actions[chosen_index]

    def _extract_ckg_references(self, forecasts: Dict) -> Dict:
        """Extract all CKG references from forecasts for explanation tracing."""
        references = {}
        for action, forecast in forecasts.items():
            if forecast['ckg_validation']:
                references[action] = {
                    'applied_rules': forecast['ckg_validation'].get('applied_rules', []),
                    'violated_rules': forecast['ckg_validation'].get('violated_rules', []),
                    'confidence': forecast['ckg_validation'].get('confidence', 0.5)
                }
        return references

# Example usage function
def demonstrate_asreh_algorithm():
    """Demonstrate the ASREH algorithm with a simple example."""
    print("=== ASREH Algorithm Demonstration ===")
    
    # Initialize components (in practice, these would be properly configured)
    ckg = ConceptualKnowledgeGraph()
    web_access = WebAccess(ckg)
    model = ASREHModel(ckg=ckg, web_access=web_access)
    sswm = SSWM()
    arlc = ARLCController(strategic_planner=None, sswm=sswm, ckg=ckg)
    em = ExplainabilityModule(model=model, sswm=sswm, ckg=ckg)
    
    # Create algorithm instance
    algorithm = ASREHAlgorithm(model, sswm, arlc, em, ckg, web_access)
    
    # Create a mock state (e.g., Tetris board)
    mock_state = torch.randn(1, 1, 20, 10)  # Batch, channels, height, width
    
    # Execute algorithm
    action, explanation, context = algorithm.execute_algorithm_step(
        mock_state, 'tetris'
    )
    
    print(f"\nChosen Action: {action}")
    print(f"\nExplanation:\n{explanation}")
    print(f"\nFull context keys: {list(context.keys())}")

if __name__ == "__main__":
    demonstrate_asreh_algorithm()