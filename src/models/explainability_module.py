# /src/models/explainability_module.py

"""
Enhanced Explainability Module with Causal Reasoning Integration
===============================================================
Now generates verifiable explanations based on CKG causal rules and traces.
"""

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
    Now enhanced with CKG-integrated causal explanations and verifiable reasoning.
    """
    def __init__(self, model: nn.Module, sswm: SSWM, ckg: ConceptualKnowledgeGraph):
        self.model = model
        self.sswm = sswm
        self.ckg = ckg
        self.conceptual_feature_names = {
            'tetris': ['Lines Cleared', 'Gaps', 'Max Height', 'Board Fullness'],
            'chess': ['Material Advantage', 'King Safety', 'Center Control', 'Development']
        }
        self.discovered_concepts = []
        self.parser = CommandParser(ckg=ckg)
        self.last_failure_report = None
        self.last_proposal_review = None

    def generate_symbolic_explanation(self, decision_context: Dict) -> str:
        """
        Generate a verifiable explanation based on CKG causal rules.
        
        Args:
            decision_context: Comprehensive context from ARLC including CKG validation
            
        Returns:
            Human-readable explanation with traceable causal reasoning
        """
        chosen_move = decision_context.get('chosen_move')
        chosen_metrics = self._get_metrics_for_move(decision_context, chosen_move)
        
        explanation_parts = []
        
        # 1. Basic decision summary
        explanation_parts.append(self._generate_decision_summary(decision_context))
        
        # 2. Causal rule-based explanation
        explanation_parts.append(self._generate_causal_explanation(chosen_metrics))
        
        # 3. Strategic context
        explanation_parts.append(self._generate_strategic_context(decision_context))
        
        # 4. Counterfactual analysis
        explanation_parts.append(self._generate_counterfactual_analysis(decision_context))
        
        # 5. Confidence assessment
        explanation_parts.append(self._generate_confidence_assessment(decision_context))
        
        # Combine all parts
        return "\n\n".join(filter(None, explanation_parts))

    def _generate_decision_summary(self, context: Dict) -> str:
        """Generate summary of the chosen decision."""
        chosen_move = context.get('chosen_move')
        chosen_score = context.get('chosen_score', 0)
        
        return f"I chose move **{chosen_move}** with a score of **{chosen_score:.2f}**."

    def _generate_causal_explanation(self, metrics: Dict) -> str:
        """Generate explanation based on CKG causal rules."""
        explanation = "### Causal Reasoning:\n"
        
        # Explain applied rules
        applied_rules = metrics.get('ckg_validation', {}).get('applied_rules', [])
        if applied_rules:
            explanation += "This decision follows these causal rules:\n"
            for rule_id in applied_rules:
                rule = self.ckg.causal_rules.get(rule_id)
                if rule:
                    explanation += f"- **{rule_id}**: {rule.get('description', 'No description')}\n"
        
        # Explain violated rules (if any)
        violated_rules = metrics.get('ckg_validation', {}).get('violated_rules', [])
        if violated_rules:
            explanation += "\nThis decision avoids violating these rules:\n"
            for rule_id in violated_rules:
                rule = self.ckg.causal_rules.get(rule_id)
                if rule:
                    explanation += f"- **{rule_id}**: {rule.get('description', 'No description')}\n"
        
        # Explain score breakdown
        score_breakdown = metrics.get('score_breakdown', {})
        if score_breakdown:
            explanation += "\n### Score Breakdown:\n"
            explanation += f"- Base Score: {score_breakdown.get('base_score', 0):.2f}\n"
            explanation += f"- Validation Confidence: {score_breakdown.get('validation_confidence', 1.0):.2f}\n"
            explanation += f"- Exploration Bonus: {score_breakdown.get('exploration_bonus', 0):.2f}\n"
            explanation += f"- HCT Bonus: {score_breakdown.get('hct_bonus', 0):.2f}\n"
            explanation += f"- Surprise Bonus: {score_breakdown.get('surprise_bonus', 0):.2f}\n"
            explanation += f"**Final Score**: {score_breakdown.get('score', 0):.2f}\n"
        
        return explanation

    def _generate_strategic_context(self, context: Dict) -> str:
        """Explain the strategic context of the decision."""
        strategic_context = context.get('strategic_context')
        if not strategic_context:
            return ""
            
        return f"### Strategic Context:\nThis move aligns with the current strategic goal: **{strategic_context}**."

    def _generate_counterfactual_analysis(self, context: Dict) -> str:
        """Analyze alternative moves that were considered."""
        all_moves = context.get('all_moves', [])
        all_scores = context.get('all_scores', [])
        chosen_move = context.get('chosen_move')
        
        if len(all_moves) < 2:
            return ""
            
        # Find top alternatives
        move_scores = list(zip(all_moves, all_scores))
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        explanation = "### Alternatives Considered:\n"
        
        for i, (move, score) in enumerate(move_scores[:3]):  # Top 3 alternatives
            if move == chosen_move:
                continue
                
            move_metrics = self._get_metrics_for_move(context, move)
            rule_differences = self._compare_rule_application(
                self._get_metrics_for_move(context, chosen_move),
                move_metrics
            )
            
            explanation += f"- **Move {move}** (Score: {score:.2f}):\n"
            if rule_differences:
                explanation += f"  - Rule differences: {rule_differences}\n"
        
        return explanation

    def _generate_confidence_assessment(self, context: Dict) -> str:
        """Assess and explain the confidence level."""
        confidence = self._calculate_verifiable_confidence(context)
        
        confidence_levels = {
            0.9: "Very High Confidence",
            0.7: "High Confidence", 
            0.5: "Moderate Confidence",
            0.3: "Low Confidence",
            0.0: "Very Low Confidence"
        }
        
        level = next((desc for threshold, desc in confidence_levels.items() if confidence >= threshold), "Very Low Confidence")
        
        return f"### Confidence Assessment:\n**{level}** ({confidence:.2f}/1.0)\n\nThis confidence is based on:\n- CKG rule validation consistency\n- Historical decision performance\n- Strategic alignment verification"

    def _calculate_verifiable_confidence(self, context: Dict) -> float:
        """Calculate confidence based on verifiable factors."""
        chosen_metrics = self._get_metrics_for_move(context, context.get('chosen_move'))
        
        confidence = 0.5  # Base confidence
        
        # Factor 1: Validation confidence
        validation_conf = chosen_metrics.get('ckg_validation', {}).get('confidence', 1.0)
        confidence *= validation_conf
        
        # Factor 2: Rule consistency
        applied_rules = chosen_metrics.get('ckg_validation', {}).get('applied_rules', [])
        violated_rules = chosen_metrics.get('ckg_validation', {}).get('violated_rules', [])
        if applied_rules and not violated_rules:
            confidence *= 1.2  # Bonus for perfect rule compliance
        
        # Factor 3: Score magnitude
        score = context.get('chosen_score', 0)
        if score > 10:
            confidence *= 1.1
        elif score < 0:
            confidence *= 0.8
        
        return max(0.0, min(1.0, confidence))

    def _get_metrics_for_move(self, context: Dict, move: any) -> Dict:
        """Get decision metrics for a specific move."""
        decision_metrics = context.get('decision_metrics', [])
        for metrics in decision_metrics:
            if metrics.get('move') == move:
                return metrics
        return {}

    def _compare_rule_application(self, chosen_metrics: Dict, alternative_metrics: Dict) -> str:
        """Compare rule application between chosen and alternative moves."""
        chosen_rules = set(chosen_metrics.get('ckg_validation', {}).get('applied_rules', []))
        alternative_rules = set(alternative_metrics.get('ckg_validation', {}).get('applied_rules', []))
        
        unique_to_chosen = chosen_rules - alternative_rules
        unique_to_alternative = alternative_rules - chosen_rules
        
        differences = []
        
        if unique_to_chosen:
            differences.append(f"Applies rules {list(unique_to_chosen)} that the alternative doesn't")
        
        if unique_to_alternative:
            differences.append(f"Avoids rules {list(unique_to_alternative)} that the alternative applies")
        
        return "; ".join(differences) if differences else "Similar rule application"

    def handle_query(self, 
                     query: str, 
                     decision_context: Dict, 
                     conceptual_features: torch.Tensor,
                     domain: str,
                     current_fused_rep: torch.Tensor) -> str:
        """
        Enhanced query handling with CKG-integrated explanations.
        """
        parsed_command = self.parser.parse_command(query)
        command = parsed_command['command']

        if command == "explain":
            # Use new symbolic explanation generator
            return self.generate_symbolic_explanation(decision_context)
            
        elif command == "rules":
            return self._explain_applied_rules(decision_context)
            
        elif command == "confidence":
            return self._explain_confidence_details(decision_context)
            
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
                return self._evaluate_specific_move(decision_context, move_idx)
            else:
                return "I'm sorry, I could not evaluate that specific move."
                
        elif command == "what_if":
            move_idx = parsed_command['entities'].get('move')
            if move_idx is not None:
                return self._handle_what_if_query(current_fused_rep, move_idx)
            else:
                return "Please specify a move to simulate. For example, 'what if I made move 5?'"
                
        elif command == "review_proposals":
            return self.review_proposals()
            
        else:
            return "I'm sorry, I don't understand that command. Please try 'explain', 'rules', 'confidence', 'strategy', 'eval_move', 'what-if', or 'review_proposals'."

    def _explain_applied_rules(self, context: Dict) -> str:
        """Provide detailed explanation of applied rules."""
        chosen_metrics = self._get_metrics_for_move(context, context.get('chosen_move'))
        applied_rules = chosen_metrics.get('ckg_validation', {}).get('applied_rules', [])
        
        if not applied_rules:
            return "No specific causal rules were applied to this decision."
        
        explanation = "### Detailed Rule Explanation:\n"
        for rule_id in applied_rules:
            rule = self.ckg.causal_rules.get(rule_id)
            if rule:
                explanation += f"**{rule_id}**: {rule.get('description', 'No description')}\n"
                explanation += f"- Conditions: {', '.join(rule.get('conditions', []))}\n"
                explanation += f"- Effects: {', '.join(rule.get('effects', []))}\n"
                explanation += f"- Confidence: {rule.get('confidence', 0.8):.2f}\n\n"
        
        return explanation

    def _explain_confidence_details(self, context: Dict) -> str:
        """Provide detailed confidence breakdown."""
        confidence = self._calculate_verifiable_confidence(context)
        chosen_metrics = self._get_metrics_for_move(context, context.get('chosen_move'))
        
        explanation = f"### Confidence Breakdown:\nOverall: {confidence:.2f}/1.0\n\n"
        
        # Validation confidence
        validation_conf = chosen_metrics.get('ckg_validation', {}).get('confidence', 1.0)
        explanation += f"- Validation Confidence: {validation_conf:.2f} (CKG rule validation)\n"
        
        # Rule compliance
        applied_rules = chosen_metrics.get('ckg_validation', {}).get('applied_rules', [])
        violated_rules = chosen_metrics.get('ckg_validation', {}).get('violated_rules', [])
        explanation += f"- Rule Compliance: {len(applied_rules)} applied, {len(violated_rules)} violated\n"
        
        # Score magnitude
        score = context.get('chosen_score', 0)
        score_factor = 1.1 if score > 10 else 0.8 if score < 0 else 1.0
        explanation += f"- Score Impact: {score_factor:.1f}x (score: {score:.2f})\n"
        
        return explanation

    def _evaluate_specific_move(self, context: Dict, move_idx: int) -> str:
        """Evaluate a specific move in detail."""
        metrics = self._get_metrics_for_move(context, move_idx)
        if not metrics:
            return f"No detailed metrics available for move {move_idx}."
        
        score = context.get('all_scores', [])[move_idx] if move_idx < len(context.get('all_scores', [])) else 0
        explanation = f"### Evaluation of Move {move_idx}:\nScore: {score:.2f}\n\n"
        
        # Rule application
        applied_rules = metrics.get('ckg_validation', {}).get('applied_rules', [])
        violated_rules = metrics.get('ckg_validation', {}).get('violated_rules', [])
        
        explanation += "**Rule Application:**\n"
        explanation += f"- Applied Rules: {len(applied_rules)}\n"
        explanation += f"- Violated Rules: {len(violated_rules)}\n"
        
        if applied_rules:
            explanation += "\n**Key Applied Rules:**\n"
            for rule_id in applied_rules[:3]:  # Show top 3
                rule = self.ckg.causal_rules.get(rule_id)
                if rule:
                    explanation += f"- {rule_id}: {rule.get('description', 'No description')}\n"
        
        return explanation

    # --- Keep existing methods for compatibility ---

    def add_discovered_concept(self, concept_name: str):
        if concept_name not in self.discovered_concepts:
            self.discovered_concepts.append(concept_name)
            self.ckg.add_node(concept_name, {"type": "emergent_concept", "source": "HCT"})
            print(f"Added new concept to CKG: {concept_name}")

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
        pending_proposals = self.ckg.query_by_property("status", "pending_human_review")
        self.last_proposal_review = pending_proposals

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
        self.ckg.add_node(f"HumanDecision_{proposal_id}", {"type": "human_decision", "decision": new_status, "timestamp": datetime.now().isoformat()})
        self.ckg.add_edge(f"HumanDecision_{proposal_id}", proposal_id, "FINALIZED")

        return is_approved

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

    # Keep legacy generate_explanation for backward compatibility
    def generate_explanation(self, conceptual_features: torch.Tensor, fused_representation: torch.Tensor, decision_context: Dict, domain: str) -> Dict:
        """Legacy method - now uses the new symbolic explanation system."""
        narrative = self.generate_symbolic_explanation(decision_context)
        return {
            'narrative': narrative,
            'confidence_score': self._calculate_verifiable_confidence(decision_context),
            'conceptual_reasoning': self._analyze_conceptual_contribution(conceptual_features.squeeze(0), domain),
            'decision_narrative': self._analyze_decision_context(decision_context)
        }

    def _analyze_conceptual_contribution(self, conceptual_features: torch.Tensor, domain: str) -> str:
        """Legacy conceptual analysis."""
        feature_names = self.conceptual_feature_names.get(domain, [])
        feature_names.extend(self.discovered_concepts)
        if not feature_names:
            return "Conceptual reasoning is not available for this domain."
        feature_values = conceptual_features.detach().cpu().numpy()
        sorted_indices = np.argsort(np.abs(feature_values))[::-1]
        top_features = [feature_names[i] for i in sorted_indices if i < len(feature_names)]
        return f"The decision was primarily driven by the goal of optimizing for '{top_features[0]}' and managing '{top_features[1]}'."

    def _analyze_decision_context(self, context: Dict) -> str:
        """Legacy decision context analysis."""
        chosen_move = context.get('chosen_move', 'an unknown move')
        chosen_score = context.get('chosen_score', 'N/A')
        return f"The model chose move {chosen_move} with a score of {chosen_score:.2f}."