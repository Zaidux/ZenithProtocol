# /src/models/explainability_module.py

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import torch.nn.functional as F
from ..nlp.command_parser import CommandParser
from .sswm import SSWM # Import the new SSWM

class ExplainabilityModule:
    """
    The Explainability Module (EM) is the "Mouth" of the system.
    It provides a clear, concise justification for the model's decision.
    """
    def __init__(self, model: nn.Module, sswm: SSWM):
        self.model = model
        self.sswm = sswm # New: Pass the SSWM to the EM
        self.conceptual_feature_names = {
            'tetris': ['Lines Cleared', 'Gaps', 'Max Height', 'Board Fullness'],
            'chess': ['Material Advantage', 'White King Safety', 'Black King Safety', 'White Center Control', 'Black Center Control']
        }
        self.discovered_concepts = [] # List to hold names of HCT concepts
        self.parser = CommandParser()

    def add_discovered_concept(self, concept_name: str):
        """Adds a newly discovered concept to the explanation vocabulary."""
        if concept_name not in self.discovered_concepts:
            self.discovered_concepts.append(concept_name)

    def handle_query(self, 
                     query: str, 
                     decision_context: Dict, 
                     conceptual_features: torch.Tensor,
                     domain: str,
                     current_fused_rep: torch.Tensor) -> str:
        """
        Handles a natural language query and provides a specific explanation.
        """
        parsed_command = self.parser.parse_command(query)
        command = parsed_command['command']

        if command == "explain":
            chosen_move = decision_context.get('chosen_move', 'an unknown move')
            chosen_score = decision_context.get('chosen_score', 'N/A')
            return f"The model chose move {chosen_move} because it had the highest conceptual score of {chosen_score:.2f}. " + self._formulate_narrative(decision_context)

        elif command == "strategy":
            conceptual_contribution = self._analyze_conceptual_contribution(
                conceptual_features.squeeze(0),
                domain
            )
            return f"The model's current long-term strategy is based on these conceptual goals: {conceptual_contribution}"

        elif command == "eval_move":
            move_idx = parsed_command['entities'].get('move')
            if move_idx is not None and 'all_scores' in decision_context and move_idx < len(decision_context['all_scores']):
                score = decision_context['all_scores'][move_idx]
                return f"Move {move_idx} was evaluated with a score of {score:.2f}."
            else:
                return "I'm sorry, I could not evaluate that specific move."

        elif command == "what_if":
            move_idx = parsed_command['entities'].get('move')
            if move_idx is not None:
                return self._handle_what_if_query(current_fused_rep, move_idx)
            else:
                return "Please specify a move to simulate. For example, 'what if I made move 5?'"

        else:
            return "I'm sorry, I don't understand that command. Please try 'explain', 'strategy', 'eval_move', or 'what-if'."

    def _handle_what_if_query(self, current_fused_rep: torch.Tensor, move_idx: int) -> str:
        """
        Simulates a hypothetical move and explains the predicted outcome.
        """
        try:
            _, predicted_reward = self.sswm.simulate_what_if_scenario(
                start_state_rep=current_fused_rep,
                hypothetical_move=move_idx,
                num_steps=1
            )
            
            # Formulate the explanation based on the predicted reward
            if predicted_reward > 0.5:
                return f"If you make move {move_idx}, the SSWM predicts a highly favorable outcome with a predicted reward of {predicted_reward:.2f}. This is a strong move."
            elif predicted_reward < -0.5:
                return f"If you make move {move_idx}, the SSWM predicts a poor outcome with a predicted penalty of {abs(predicted_reward):.2f}. This move could lead to a significant disadvantage."
            else:
                return f"The SSWM predicts a neutral outcome for move {move_idx}, with a predicted reward of {predicted_reward:.2f}. There might be a better strategic option."
        
        except Exception as e:
            return f"An error occurred during the simulation: {e}"


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

        moe_context = decision_context.get('moe_context', {})
        moe_explanation = self._analyze_moe_contribution(moe_context)
        explanation['moe_reasoning'] = moe_explanation

        eom_explanation = self._analyze_eom_contribution(decision_context)
        explanation['eom_reasoning'] = eom_explanation

        final_explanation = self._formulate_narrative(explanation)
        explanation['narrative'] = final_explanation

        return explanation

    def _analyze_conceptual_contribution(self, conceptual_features: torch.Tensor, domain: str) -> str:
        """
        Provides a simplified explanation based on the most influential conceptual features.
        """
        feature_names = self.conceptual_feature_names.get(domain, [])
        feature_names.extend(self.discovered_concepts)

        if not feature_names:
            return "Conceptual reasoning is not available for this domain."

        feature_values = conceptual_features.detach().cpu().numpy()
        sorted_indices = np.argsort(np.abs(feature_values))[::-1]
        top_features = [feature_names[i] for i in sorted_indices if i < len(feature_names)]

        if not top_features:
            return "No strong conceptual features detected."

        if domain == 'tetris':
            return (f"The decision was primarily driven by the goal of minimizing '{top_features[1]}' and "
                    f"optimizing for '{top_features[0]}'.")
        elif domain == 'chess':
            material_val = feature_values[feature_names.index('Material Advantage')] if 'Material Advantage' in feature_names else 0
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

    def _analyze_moe_contribution(self, moe_context: Dict) -> str:
        """
        Explains which MoE expert was chosen and why.
        """
        if not moe_context:
            return "MoE analysis is not available."

        expert_idx = moe_context.get('chosen_expert', 'N/A')
        confidence = moe_context.get('confidence', 'N/A')

        if expert_idx != 'N/A':
            return f"The model's router selected expert #{expert_idx} with a confidence score of {confidence:.2f}, indicating its specialization for this type of problem."
        else:
            return "The model's router was unable to select a clear expert."

    def _analyze_eom_contribution(self, decision_context: Dict) -> str:
        """
        Explains the conceptual energy of the chosen move.
        """
        eom_bonus = decision_context.get('eom_bonus', 0.0)
        eom_text = ""
        if eom_bonus > 5.0:
            eom_text = "The model's chosen move generated a very high conceptual energy, indicating a significant and strategic shift in the board's state."
        elif eom_bonus > 1.0:
            eom_text = "The move resulted in a moderate conceptual energy, suggesting a meaningful but not groundbreaking change."
        else:
            eom_text = "The move generated low conceptual energy, indicating a minor, tactical adjustment to the board."
        return eom_text

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
        moe_text = explanation.get('moe_reasoning', '')
        eom_text = explanation.get('eom_reasoning', '')
        decision_text = explanation.get('decision_narrative', '')

        return f"{conceptual_text} {moe_text} {eom_text} {decision_text}"
