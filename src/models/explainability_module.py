# /src/models/explainability_module.py

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch.nn.functional as F
from ..nlp.command_parser import CommandParser
from .sswm import SSWM
from ..conceptual_knowledge_graph.ckg import ConceptualKnowledgeGraph
import hashlib
import json
from datetime import datetime

class ExplainabilityModule:
    """
    The Explainability Module (EM) is the "Mouth" of the system.
    It now includes a critical interface for human-in-the-loop governance.
    """
    def __init__(self, model: nn.Module, sswm: SSWM, ckg: ConceptualKnowledgeGraph):
        self.model = model
        self.sswm = sswm
        self.ckg = ckg
        self.conceptual_feature_names = {
            'tetris': ['Lines Cleared', 'Gaps', 'Max Height', 'Board Fullness'],
            'chess': ['Material Advantage', 'White King Safety', 'Black King Safety', 'White Center Control', 'Black Center Control']
        }
        self.discovered_concepts = []
        self.parser = CommandParser(ckg=ckg)
        self.last_failure_report = None
        self.last_proposal_review = None

    def add_discovered_concept(self, concept_name: str):
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
        elif command == "review_proposals": # New Command
            return self.review_proposals()
        else:
            return "I'm sorry, I don't understand that command. Please try 'explain', 'strategy', 'eval_move', 'what-if', or 'review_proposals'."

    def _handle_what_if_query(self, current_fused_rep: torch.Tensor, move_idx: int) -> str:
        try:
            _, predicted_reward = self.sswm.simulate_what_if_scenario(
                start_state_rep=current_fused_rep,
                hypothetical_move=move_idx,
                num_steps=1
            )
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

    def review_proposals(self) -> str:
        """
        Queries the CKG for all pending architectural proposals and presents them for review.
        """
        pending_proposals = self.ckg.query_by_property("status", "pending_human_review")
        self.last_proposal_review = pending_proposals # Store for later use

        if not pending_proposals:
            return "There are no architectural proposals pending your review."

        output = "Pending Architectural Proposals:\n"
        for proposal in pending_proposals:
            proposal_id = proposal['node']['proposal_id']
            proposer = proposal['node']['proposer']
            upgrade_type = proposal['node']['type']
            predicted_impact = proposal['node']['predicted_impact'].get('impact', 'N/A')
            reasoning = proposal['node']['predicted_impact'].get('reason', 'N/A')
            
            output += f"\n- **Proposal ID:** {proposal_id}\n"
            output += f"  - **Proposer:** {proposer}\n"
            output += f"  - **Type:** {upgrade_type}\n"
            output += f"  - **Predicted Impact:** {predicted_impact}\n"
            output += f"  - **Reasoning:** {reasoning}\n"
            output += f"  - **To approve, say:** 'Approve proposal {proposal_id}'\n"
            output += f"  - **To reject, say:** 'Reject proposal {proposal_id}'\n"

        return output

    def get_human_confirmation(self, proposal_id: str, is_approved: bool) -> bool:
        """
        This method represents the human-in-the-loop decision point.
        It updates the proposal's status in the CKG based on human input.
        """
        proposal = self.ckg.query(proposal_id)
        if not proposal:
            print(f"Proposal with ID '{proposal_id}' not found.")
            return False

        if is_approved:
            new_status = "approved"
            print(f"Proposal '{proposal_id}' has been approved by human.")
        else:
            new_status = "rejected"
            print(f"Proposal '{proposal_id}' has been rejected by human.")

        self.ckg.update_node_properties(proposal_id, {"status": new_status, "human_decision": new_status})
        
        # Log the human-in-the-loop decision to the CKG
        self.ckg.add_node(f"HumanDecision_{proposal_id}", {"type": "human_decision", "decision": new_status, "timestamp": datetime.now().isoformat()})
        self.ckg.add_edge(f"HumanDecision_{proposal_id}", proposal_id, "FINALIZED")
        
        return is_approved

    def generate_explanation(self, conceptual_features: torch.Tensor, fused_representation: torch.Tensor, decision_context: Dict, domain: str) -> Dict:
        explanation = {}
        conceptual_contribution = self._analyze_conceptual_contribution(conceptual_features.squeeze(0), domain)
        explanation['conceptual_reasoning'] = conceptual_contribution
        decision_narrative = self._analyze_decision_context(decision_context)
        explanation['decision_narrative'] = decision_narrative
        moe_context = decision_context.get('moe_context', {})
        moe_explanation = self._analyze_moe_contribution(moe_context)
        explanation['moe_reasoning'] = moe_explanation
        eom_explanation = self._analyze_eom_contribution(decision_context)
        explanation['eom_reasoning'] = eom_explanation
        counterfactual_narrative = self._analyze_counterfactuals(decision_context, fused_representation)
        explanation['counterfactual_reasoning'] = counterfactual_narrative
        confidence_score = self._get_confidence_score(decision_context)
        explanation['confidence_score'] = confidence_score
        final_explanation = self._formulate_narrative(explanation)
        explanation['narrative'] = final_explanation
        return explanation

    def _get_confidence_score(self, decision_context: Dict) -> float:
        chosen_score = decision_context.get('chosen_score', 0)
        confidence_score = (chosen_score / max(decision_context.get('all_scores', [chosen_score]))) if decision_context.get('all_scores') else 0.5
        conceptual_factors = decision_context.get('conceptual_factors', [])
        for concept in conceptual_factors:
            node = self.ckg.query(concept)
            if node:
                if node['node'].get('source') == 'training_data' or node['node'].get('verifiability_score', 0) > 0.8:
                    confidence_score += 0.1
                elif node['node'].get('source') == 'HCT':
                    confidence_score -= 0.1
        if decision_context.get('counterfactuals'):
            confidence_score += 0.05
        chosen_move_id = f"Move_{decision_context.get('chosen_move')}"
        verifiable_record = self.ckg.get_verifiable_record(chosen_move_id)
        if verifiable_record:
            confidence_score += 0.2
        return max(0.0, min(1.0, confidence_score))

    def _analyze_conceptual_contribution(self, conceptual_features: torch.Tensor, domain: str) -> str:
        feature_names = self.conceptual_feature_names.get(domain, [])
        feature_names.extend(self.discovered_concepts)
        if not feature_names:
            return "Conceptual reasoning is not available for this domain."
        feature_values = conceptual_features.detach().cpu().numpy()
        sorted_indices = np.argsort(np.abs(feature_values))[::-1]
        top_features = [feature_names[i] for i in sorted_indices if i < len(feature_names)]
        conceptual_info = self.ckg.query('conceptual_reasoning')
        conceptual_reasoning_text = conceptual_info['node'].get('content', 'The model is analyzing abstract properties of the state.')
        return f"{conceptual_reasoning_text} The decision was primarily driven by the goal of optimizing for '{top_features[0]}' and managing '{top_features[1]}'."

    def _analyze_moe_contribution(self, moe_context: Dict) -> str:
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
        eom_bonus = decision_context.get('eom_bonus', 0.0)
        eom_text = ""
        if eom_bonus > 5.0:
            eom_text = self.ckg.query("high_eom_change")['node'].get('description', 'a very high conceptual energy, indicating a significant and strategic shift.')
        elif eom_bonus > 1.0:
            eom_text = self.ckg.query("moderate_eom_change")['node'].get('description', 'a moderate conceptual energy, suggesting a meaningful but not groundbreaking change.')
        else:
            eom_text = self.ckg.query("low_eom_change")['node'].get('description', 'low conceptual energy, indicating a minor, tactical adjustment.')
        return f"The move generated {eom_text}."

    def _analyze_counterfactuals(self, decision_context: Dict, current_fused_rep: torch.Tensor) -> str:
        chosen_move = decision_context.get('chosen_move')
        all_scores = decision_context.get('all_scores', [])
        if not all_scores or len(all_scores) < 2:
            return "Counterfactual analysis not available."
        sorted_scores = sorted(zip(all_scores, range(len(all_scores))), reverse=True)
        best_rejected_move_info = None
        for score, move_idx in sorted_scores:
            if move_idx != chosen_move:
                best_rejected_move_info = (score, move_idx)
                break
        if not best_rejected_move_info:
            return "Counterfactual analysis not available."
        best_rejected_score, best_rejected_move = best_rejected_move_info
        sim_outcomes = self.sswm.simulate_multiple_what_if_scenarios(
            start_state_rep=current_fused_rep,
            hypothetical_moves=[chosen_move, best_rejected_move],
            num_steps=1
        )
        chosen_reward = sim_outcomes[chosen_move][1]
        rejected_reward = sim_outcomes[best_rejected_move][1]
        chosen_outcome_desc = "a positive outcome" if chosen_reward > 0 else "a neutral outcome"
        rejected_outcome_desc = "a positive outcome" if rejected_reward > 0 else "a neutral outcome"
        narrative = (
            f"While move {best_rejected_move} also had a high score, the SSWM predicted that your chosen move, {chosen_move}, "
            f"would lead to a more favorable outcome with a reward of {chosen_reward:.2f} compared to {rejected_reward:.2f}. "
            f"The chosen move was selected to prioritize a more strategic long-term goal."
        )
        return narrative

    def _analyze_decision_context(self, context: Dict) -> str:
        chosen_move = context.get('chosen_move', 'an unknown move')
        chosen_score = context.get('chosen_score', 'N/A')
        all_scores = context.get('all_scores', [])
        best_possible_score = max(all_scores) if all_scores else 'N/A'
        if chosen_score == best_possible_score:
            return f"The model chose the best possible move, which had the highest calculated score of {chosen_score:.2f}."
        else:
            return f"The model chose a move with a score of {chosen_score:.2f}, even though a move with a slightly higher score of {best_possible_score:.2f} was available. This suggests the model was in an exploratory mode to discover new concepts or patterns, which is a key part of the learning process."

    def _formulate_narrative(self, explanation: Dict) -> str:
        confidence_text = f"The model's confidence in this explanation is {explanation['confidence_score']:.2f}. "
        conceptual_text = explanation.get('conceptual_reasoning', '')
        moe_text = explanation.get('moe_reasoning', '')
        eom_text = explanation.get('eom_reasoning', '')
        decision_text = explanation.get('decision_narrative', '')
        counterfactual_text = explanation.get('counterfactual_reasoning', '')
        return f"{confidence_text} {conceptual_text} {moe_text} {eom_text} {decision_text} {counterfactual_text}"

    def analyze_and_report_failure(self, 
                                   original_input: torch.Tensor, 
                                   adversarial_input: torch.Tensor,
                                   original_output: torch.Tensor,
                                   adversarial_output: torch.Tensor) -> Dict:
        output_divergence = F.mse_loss(original_output, adversarial_output).item()
        input_change = F.mse_loss(original_input, adversarial_input).item()
        if input_change < 0.1 and output_divergence > 0.5:
            error_type = "conceptual_misinterpretation"
            explanation = "A small, unnoticeable change in the conceptual features caused a large change in the output, indicating a fundamental misunderstanding or hallucination."
            causal_factors = ["conceptual_misinterpretation"]
        else:
            error_type = "visual_hallucination"
            explanation = "The model was unable to correctly interpret the visual input due to a significant perturbation, leading to a hallucinated output."
            causal_factors = ["visual_hallucination", "adversarial_input"]
        failure_report = {
            "type": error_type,
            "explanation": explanation,
            "causal_factors": causal_factors,
            "metrics": {
                "output_divergence": output_divergence,
                "input_change": input_change
            }
        }
        report_hash = hashlib.sha256(str(failure_report).encode()).hexdigest()
        self.ckg.add_node(f"Adversarial_Failure_{report_hash}", {
            "type": "adversarial_failure",
            "error_type": error_type,
            "explanation": explanation,
            "input_change": input_change,
            "output_divergence": output_divergence,
        })
        self.ckg.add_edge("AdversarialModule", f"Adversarial_Failure_{report_hash}", "CAUSED")
        self.last_failure_report = failure_report
        return failure_report

    def get_last_failure_report(self) -> Dict | None:
        return self.last_failure_report
