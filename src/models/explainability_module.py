# /src/modules/explainability_module.py

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple

class ExplainabilityModule:
    def __init__(self, model):
        self.model = model
        self.conceptual_feature_names = {
            'chess': ['Material Advantage', 'White King Safety', 'Black King Safety', 'White Center Control', 'Black Center Control'],
            'tetris': ['Lines Cleared', 'Gaps', 'Max Height', 'Board Fullness']
        }
        
    def generate_explanation(self, conceptual_embedding, attention_weights, domain: str) -> Dict:
        """
        Generates a human-readable explanation based on the model's internal state.
        """
        explanation = {}
        
        # 1. Analyze Conceptual Layer's Contribution
        conceptual_contribution = self._analyze_conceptual_contribution(conceptual_embedding, domain)
        explanation['conceptual_reasoning'] = conceptual_contribution
        
        # 2. Analyze Attention Layer's Focus
        attention_focus = self._analyze_attention_focus(attention_weights)
        explanation['visual_focus'] = attention_focus
        
        # 3. Formulate the Final Explanation
        final_explanation = self._formulate_narrative(explanation, domain)
        explanation['narrative'] = final_explanation
        
        return explanation

    def _analyze_conceptual_contribution(self, conceptual_embedding, domain: str) -> str:
        """
        Maps conceptual embeddings to a verbal explanation.
        This is a simplified approach; in a real-world system, this would be more complex.
        """
        embedding_np = conceptual_embedding.detach().cpu().numpy().squeeze()
        
        top_concepts = np.argsort(embedding_np)[::-1]
        
        # Dynamically get the top 2 most influential concepts
        concept_names = self.conceptual_feature_names.get(domain, [])
        if len(concept_names) >= 2:
            top1_name = concept_names[top_concepts[0]]
            top2_name = concept_names[top_concepts[1]]
            return f"The model's decision was primarily influenced by the goal of improving {top1_name} and {top2_name}."
        else:
            return "The model's conceptual reasoning is not yet clear."
        
    def _analyze_attention_focus(self, attention_weights) -> str:
        """
        Interprets the attention weights to describe what part of the board the model focused on.
        """
        weights_np = attention_weights.detach().cpu().numpy().squeeze()
        # Find the coordinates with the highest attention weight
        if len(weights_np.shape) > 1:
            max_idx = np.argmax(weights_np)
            # Assuming a flattened grid, convert index back to a position
            # This would need to be a more robust mapping for different grid sizes
            return f"The model focused its attention on a specific area of the board to make the decision."
        else:
            return "Attention focus could not be determined from the weights."
            
    def _formulate_narrative(self, explanation: Dict, domain: str) -> str:
        """
        Combines the different analytical outputs into a single narrative.
        """
        conceptual_text = explanation.get('conceptual_reasoning', '')
        visual_text = explanation.get('visual_focus', '')
        
        return f"Based on its analysis, the model's core goal was {conceptual_text}. To achieve this, it {visual_text}."
  
