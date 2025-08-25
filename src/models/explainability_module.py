# /src/models/explainability_module.py

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import torch.nn.functional as F

class ExplainabilityModule:
    """
    The Explainability Module (EM) is the "Mouth" of the system.
    It provides a clear, concise justification for the model's decision.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.conceptual_feature_names = {
            'tetris': ['Lines Cleared', 'Gaps', 'Max Height', 'Board Fullness']
        }
    
    def generate_explanation(self, 
                             conceptual_features: torch.Tensor, 
                             fused_representation: torch.Tensor,
                             decision_context: Dict,
                             domain: str) -> Dict:
        """
        Generates a human-readable explanation based on the model's internal state.
        
        Args:
            conceptual_features: The raw conceptual features used as input.
            fused_representation: The output of the Conceptual Attention layer.
            decision_context: A dictionary from the ARLCController with move details.
            domain: The current game domain (e.g., 'tetris').
        """
        explanation = {}

        # 1. Analyze Conceptual Layer's Contribution
        conceptual_contribution = self._analyze_conceptual_contribution(
            conceptual_features.squeeze(0),
            domain
        )
        explanation['conceptual_reasoning'] = conceptual_contribution

        # 2. Analyze Decision-Making Context
        decision_narrative = self._analyze_decision_context(decision_context)
        explanation['decision_narrative'] = decision_narrative
        
        # 3. Formulate the Final Explanation
        final_explanation = self._formulate_narrative(explanation)
        explanation['narrative'] = final_explanation

        return explanation

    def _analyze_conceptual_contribution(self, conceptual_features: torch.Tensor, domain: str) -> str:
        """
        Provides a simplified explanation based on the most influential conceptual features.
        This is a placeholder for a more advanced explanation that would analyze gradients.
        """
        feature_names = self.conceptual_feature_names.get(domain, [])
        if not feature_names:
            return "Conceptual reasoning is not available for this domain."
            
        feature_values = conceptual_features.detach().cpu().numpy()
        
        # Find the most influential features
        sorted_indices = np.argsort(np.abs(feature_values))[::-1]
        
        top_features = [feature_names[i] for i in sorted_indices]
        
        return (f"The decision was primarily driven by the goal of minimizing '{top_features[1]}' and "
                f"optimizing for '{top_features[0]}'.")

    def _analyze_decision_context(self, context: Dict) -> str:
        """
        Translates the ARLC's decision into a simple narrative.
        """
        chosen_move = context.get('chosen_move', 'an unknown move')
        chosen_score = context.get('chosen_score', 'N/A')
        
        all_scores = context.get('all_scores', [])
        best_possible_score = max(all_scores) if all_scores else 'N/A'
        
        if chosen_score == best_possible_score:
            return f"The model chose the best possible move, which had the highest calculated score of {chosen_score:.2f}."
        else:
            return f"The model chose a move with a score of {chosen_score:.2f}, even though a move with a slightly higher score of {best_possible_score:.2f} was available. This suggests the model was in an exploratory mode."

    def _formulate_narrative(self, explanation: Dict) -> str:
        """
        Combines the different analytical outputs into a single, cohesive narrative.
        """
        conceptual_text = explanation.get('conceptual_reasoning', '')
        decision_text = explanation.get('decision_narrative', '')
        
        return f"{conceptual_text} {decision_text}"
