# /src/models/explainability_module.py

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import torch.nn.functional as F
from ..nlp.command_parser import CommandParser
from .sswm import SSWM
from ..conceptual_knowledge_graph.ckg import ConceptualKnowledgeGraph # New Import

class ExplainabilityModule:
    """
    The Explainability Module (EM) is the "Mouth" of the system.
    It provides a clear, concise justification for the model's decision.
    It now uses the Conceptual Knowledge Graph to generate richer, more grounded explanations.
    """
    def __init__(self, model: nn.Module, sswm: SSWM, ckg: ConceptualKnowledgeGraph): # New CKG dependency
        self.model = model
        self.sswm = sswm
        self.ckg = ckg # New: CKG instance
        self.conceptual_feature_names = {
            'tetris': ['Lines Cleared', 'Gaps', 'Max Height', 'Board Fullness'],
            'chess': ['Material Advantage', 'White King Safety', 'Black King Safety', 'White Center Control', 'Black Center Control']
        }
        self.discovered_concepts = []
        self.parser = CommandParser(ckg=ckg) # Pass the CKG to the parser

    def add_discovered_concept(self, concept_name: str):
        """Adds a newly discovered concept to the explanation vocabulary and the CKG."""
        if concept_name not in self.discovered_concepts:
            self.discovered_concepts.append(concept_name)
            self.ckg.add_node(concept_name, {"type": "emergent_concept", "source": "HCT"})
            print(f"Added new concept to CKG: {concept_name}")


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
            explanation = self.generate_explanation(conceptual_features, current_fused_rep, decision_context, domain)
            return explanation['narrative']

        elif command == "strategy":
            conceptual_contribution = self._analyze_conceptual_contribution(
                conceptual_features.squeeze(0),
                domain
            )
            strategy_info = self.ckg.query('strategy')
            return f"The model's current long-term strategy is based on these conceptual goals: {conceptual_contribution}. This strategy is stored in the CKG as: {strategy_info}"

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
            # Fetch related concepts from the CKG to enrich the explanation
            predicted_outcome_concept = self.ckg.query("favorable_outcome") if predicted_reward > 0.5 else self.ckg.query("unfavorable_outcome")
            outcome_description = predicted_outcome_concept['node'].get('description', '')

            if predicted_reward > 0.5:
                return f"If you make move {move_idx}, the SSWM predicts a highly favorable outcome with a predicted reward of {predicted_reward:.2f}. {outcome_description} This is a strong move."
            elif predicted_reward < -0.5:
                return f"If you make move {move_idx}, the SSWM predicts a poor outcome with a predicted penalty of {abs(predicted_reward):.2f}. {outcome_description} This move could lead to a significant disadvantage."
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
        This version includes a confidence score based on the CKG.
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
        
        # New: Add a confidence and hallucination score
        confidence_score = self._get_confidence_score(decision_context)
        explanation['confidence_score'] = confidence_score

        final_explanation = self._formulate_narrative(explanation)
        explanation['narrative'] = final_explanation

        return explanation

    def _get_confidence_score(self, decision_context: Dict) -> float:
        """
        Calculates a confidence score for the explanation.
        Higher scores indicate more grounded reasoning.
        """
        # A simple, rule-based confidence score for now
        # In a real model, this would be based on model uncertainty.
        chosen_score = decision_context.get('chosen_score', 0)
        
        # Check if the chosen move is a known, high-value move in the CKG
        move_info = self.ckg.query(f"Move_{decision_context.get('chosen_move')}")
        is_known_good_move = move_info and 'is_high_value' in move_info['node'].get('properties', {})

        if is_known_good_move:
            return 0.95  # High confidence
        elif chosen_score > 1.0:
            return 0.75  # Good confidence
        else:
            return 0.40  # Low confidence, might be a random or exploratory move.

    def _analyze_conceptual_contribution(self, conceptual_features: torch.Tensor, domain: str) -> str:
        """
        Provides a simplified explanation based on the most influential conceptual features.
        This version adds context from the CKG.
        """
        feature_names = self.conceptual_feature_names.get(domain, [])
        feature_names.extend(self.discovered_concepts)

        if not feature_names:
            return "Conceptual reasoning is not available for this domain."

        feature_values = conceptual_features.detach().cpu().numpy()
        sorted_indices = np.argsort(np.abs(feature_values))[::-1]
        top_features = [feature_names[i] for i in sorted_indices if i < len(feature_names)]
        
        # Get conceptual explanation from the CKG
        conceptual_info = self.ckg.query('conceptual_reasoning')
        conceptual_reasoning_text = conceptual_info['node'].get('content', 'The model is analyzing abstract properties of the state.')

        return f"{conceptual_reasoning_text} The decision was primarily driven by the goal of optimizing for '{top_features[0]}' and managing '{top_features[1]}'."

    def _analyze_moe_contribution(self, moe_context: Dict) -> str:
        """
        Explains which MoE expert was chosen and why.
        """
        if not moe_context:
            return "MoE analysis is not available."

        expert_idx = moe_context.get('chosen_expert', 'N/A')
        confidence = moe_context.get('confidence', 'N/A')

        if expert_idx != 'N/A':
            expert_info = self.ckg.query(f"MoE_Expert_{expert_idx}")
            expert_desc = expert_info['node'].get('description', 'a specialized problem-solver')
            return f"The model's router selected expert #{expert_idx} with a confidence score of {confidence:.2f}, indicating its specialization for this type of problem. This expert is known for being {expert_desc}."
        else:
            return "The model's router was unable to select a clear expert."


    def _analyze_eom_contribution(self, decision_context: Dict) -> str:
        """
        Explains the conceptual energy of the chosen move.
        This version retrieves a more nuanced description from the CKG.
        """
        eom_bonus = decision_context.get('eom_bonus', 0.0)
        eom_text = ""
        
        if eom_bonus > 5.0:
            eom_text = self.ckg.query("high_eom_change")['node'].get('description', 'a very high conceptual energy, indicating a significant and strategic shift.')
        elif eom_bonus > 1.0:
            eom_text = self.ckg.query("moderate_eom_change")['node'].get('description', 'a moderate conceptual energy, suggesting a meaningful but not groundbreaking change.')
        else:
            eom_text = self.ckg.query("low_eom_change")['node'].get('description', 'low conceptual energy, indicating a minor, tactical adjustment.')
            
        return f"The move generated {eom_text}."


    def _analyze_decision_context(self, context: Dict) -> str:
        """
        Translates the ARLC's decision into a simple narrative.
        This version is slightly more verbose to demonstrate the CKG usage.
        """
        chosen_move = context.get('chosen_move', 'an unknown move')
        chosen_score = context.get('chosen_score', 'N/A')

        all_scores = context.get('all_scores', [])
        best_possible_score = max(all_scores) if all_scores else 'N/A'
        
        if chosen_score == best_possible_score:
            return f"The model chose the best possible move, which had the highest calculated score of {chosen_score:.2f}."
        else:
            return f"The model chose a move with a score of {chosen_score:.2f}, even though a move with a slightly higher score of {best_possible_score:.2f} was available. This suggests the model was in an exploratory mode to discover new concepts or patterns, which is a key part of the learning process."

    def _formulate_narrative(self, explanation: Dict) -> str:
        """
        Combines the different analytical outputs into a single, cohesive narrative.
        """
        confidence_text = f"The model's confidence in this explanation is {explanation['confidence_score']:.2f}. "
        
        conceptual_text = explanation.get('conceptual_reasoning', '')
        moe_text = explanation.get('moe_reasoning', '')
        eom_text = explanation.get('eom_reasoning', '')
        decision_text = explanation.get('decision_narrative', '')

        return f"{confidence_text} {conceptual_text} {moe_text} {eom_text} {decision_text}"
