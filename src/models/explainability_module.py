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
            'tetris': ['Lines Cleared', 'Gaps', 'Max Height', 'Board Fullness'],
            'chess': ['Material Advantage', 'White King Safety', 'Black King Safety', 'White Center Control', 'Black Center Control']
        }

    def generate_explanation(self, 
                             conceptual_features: torch.Tensor, 
                             fused_representation: torch.Tensor,
                             decision_context: Dict,
                             domain: str) -> Dict:
        """
        Generates a human-readable explanation based on the model's internal state.
        """
        explanation = {}

        conceptual_contribution = self._analyze_conceptual_contribution(
            conceptual_features.squeeze(0),
            domain
        )
        explanation['conceptual_reasoning'] = conceptual_contribution

        decision_narrative = self._analyze_decision_context(decision_context)
        explanation['decision_narrative'] = decision_narrative

        final_explanation = self._formulate_narrative(explanation)
        explanation['narrative'] = final_explanation

        return explanation

    def _analyze_conceptual_contribution(self, conceptual_features: torch.Tensor, domain: str) -> str:
        """
        Provides a simplified explanation based on the most influential conceptual features.
        """
        feature_names = self.conceptual_feature_names.get(domain, [])
        if not feature_names:
            return "Conceptual reasoning is not available for this domain."

        feature_values = conceptual_features.detach().cpu().numpy()

        # Find the two most influential features based on absolute value
        sorted_indices = np.argsort(np.abs(feature_values))[::-1]
        top_features = [feature_names[i] for i in sorted_indices]
        
        # Craft a more dynamic explanation based on the domain
        if domain == 'tetris':
            return (f"The decision was primarily driven by the goal of minimizing '{top_features[1]}' and "
                    f"optimizing for '{top_features[0]}'.")
        elif domain == 'chess':
            material_val = feature_values[feature_names.index('Material Advantage')]
            if material_val > 0:
                material_text = "The model is seeking to increase its material advantage."
            elif material_val < 0:
                material_text = "The model is seeking to mitigate its material disadvantage."
            else:
                material_text = "Material is balanced."
            
            return (f"The model analyzed the board's abstract properties. {material_text} It is also prioritizing "
                    f"'{top_features[0]}' and managing '{top_features[1]}'.")
        else:
            return "Conceptual reasoning is not available for this domain."

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
