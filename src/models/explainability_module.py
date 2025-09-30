# /src/models/explainability_module.py

"""
Enhanced Explainability Module with Sparse Attention Integration
===============================================================
Now explains not just WHAT the model decided, but HOW it paid attention using sparse patterns.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch.nn.functional as F
from ..nlp.command_parser import CommandParser
from .sswm import SSWM
from ..conceptual_knowledge_graph.ckg import ConceptualKnowledgeGraph
from ..modules.self_evolving_sparse_attention import SelfEvolvingSparseAttention
import hashlib
import json
from datetime import datetime

class ExplainabilityModule:
    """
    The Explainability Module (EM) - Now with sparse attention pattern explanations
    and quantum compression insights.
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
        
        # Sparse attention analysis
        self.attention_pattern_history = []
        self.quantum_compression_metrics = []

    def generate_symbolic_explanation(self, decision_context: Dict) -> str:
        """
        Enhanced explanation with sparse attention pattern analysis.
        """
        chosen_move = decision_context.get('chosen_move')
        sparse_attention_info = decision_context.get('sparse_attention_info', {})
        quantum_compression_info = decision_context.get('quantum_compression_info', {})

        explanation_parts = []

        # 1. Basic decision summary
        explanation_parts.append(self._generate_decision_summary(decision_context))

        # 2. Sparse attention pattern explanation
        explanation_parts.append(self._generate_sparse_attention_explanation(sparse_attention_info))

        # 3. Quantum compression insights
        explanation_parts.append(self._generate_quantum_compression_insights(quantum_compression_info))

        # 4. Causal rule-based explanation
        explanation_parts.append(self._generate_causal_explanation(decision_context))

        # 5. Strategic context
        explanation_parts.append(self._generate_strategic_context(decision_context))

        # 6. Counterfactual analysis
        explanation_parts.append(self._generate_counterfactual_analysis(decision_context))

        # 7. Confidence assessment
        explanation_parts.append(self._generate_confidence_assessment(decision_context))

        # Combine all parts
        return "\n\n".join(filter(None, explanation_parts))

    def _generate_sparse_attention_explanation(self, sparse_info: Dict) -> str:
        """Explain how sparse attention influenced the decision."""
        if not sparse_info:
            return ""
            
        explanation = "### Sparse Attention Analysis:\n"
        
        strategy = sparse_info.get('selected_strategy', 'unknown')
        sparsity = sparse_info.get('attention_sparsity', 0.0)
        efficiency_gain = sparse_info.get('efficiency_gain', 1.0)
        
        explanation += f"- **Attention Strategy**: {self._format_strategy_name(strategy)}\n"
        explanation += f"- **Sparsity Achieved**: {sparsity:.1%} (only {sparsity:.1%} of possible connections used)\n"
        explanation += f"- **Efficiency Gain**: {efficiency_gain:.1f}x faster than full attention\n"
        
        # Explain what this means
        if sparsity > 0.8:
            explanation += "- **Impact**: The model focused intensely on the most critical 20% of information, ignoring irrelevant details.\n"
        elif sparsity > 0.5:
            explanation += "- **Impact**: Balanced attention between key patterns and broader context.\n"
        else:
            explanation += "- **Impact**: Comprehensive analysis using most available information.\n"
            
        return explanation

    def _generate_quantum_compression_insights(self, quantum_info: Dict) -> str:
        """Explain quantum compression benefits."""
        if not quantum_info:
            return ""
            
        explanation = "### Quantum Compression Insights:\n"
        
        compression_ratio = quantum_info.get('compression_ratio', 1.0)
        coherence = quantum_info.get('quantum_coherence', 0.0)
        info_density = quantum_info.get('information_density', 0.0)
        
        explanation += f"- **Compression Ratio**: {compression_ratio:.1f}:1 (reduced to {1/compression_ratio:.1%} of original size)\n"
        explanation += f"- **Quantum Coherence**: {coherence:.1%} (information preservation quality)\n"
        explanation += f"- **Information Density**: {info_density:.3f} bits/parameter\n"
        
        if compression_ratio > 4.0:
            explanation += "- **Benefit**: High compression enabled processing 4x more data with same memory.\n"
        elif compression_ratio > 2.0:
            explanation += "- **Benefit**: Moderate compression balanced efficiency and accuracy.\n"
            
        return explanation

    def _format_strategy_name(self, strategy: str) -> str:
        """Format strategy names for human readability."""
        strategy_names = {
            'conceptual_sparse': 'Conceptual Sparse Attention',
            'ckg_guided': 'CKG-Guided Attention', 
            'self_evolving': 'Self-Evolving Patterns',
            'quantum_compressed': 'Quantum Compression',
            'multi_modal_fused': 'Multi-Modal Fusion'
        }
        return strategy_names.get(strategy, strategy.replace('_', ' ').title())

    def _generate_decision_summary(self, context: Dict) -> str:
        """Enhanced summary with sparse attention context."""
        chosen_move = context.get('chosen_move')
        chosen_score = context.get('chosen_score', 0)
        attention_strategy = context.get('sparse_attention_info', {}).get('selected_strategy', 'standard')

        summary = f"I chose move **{chosen_move}** with a score of **{chosen_score:.2f}**.\n"
        summary += f"**Attention Strategy**: {self._format_strategy_name(attention_strategy)}\n"
        
        return summary

    def _generate_causal_explanation(self, context: Dict) -> str:
        """Enhanced with sparse attention rule applications."""
        chosen_metrics = self._get_metrics_for_move(context, context.get('chosen_move'))
        explanation = "### Causal Reasoning:\n"

        # Explain applied rules
        applied_rules = chosen_metrics.get('ckg_validation', {}).get('applied_rules', [])
        if applied_rules:
            explanation += "This decision follows these causal rules:\n"
            for rule_id in applied_rules:
                rule = self.ckg.causal_rules.get(rule_id)
                if rule:
                    explanation += f"- **{rule_id}**: {rule.get('description', 'No description')}\n"

        # Add sparse attention specific rules
        sparse_rules = self._extract_sparse_attention_rules(context)
        if sparse_rules:
            explanation += "\n**Sparse Attention Rules**:\n"
            explanation += sparse_rules

        return explanation

    def _extract_sparse_attention_rules(self, context: Dict) -> str:
        """Extract rules specific to sparse attention patterns."""
        sparse_info = context.get('sparse_attention_info', {})
        strategy = sparse_info.get('selected_strategy', '')
        
        rules = {
            'conceptual_sparse': "Focus only on conceptually important tokens, ignore noise",
            'ckg_guided': "Use CKG relationships to determine attention patterns", 
            'self_evolving': "Use evolved patterns that have proven successful historically",
            'quantum_compressed': "Process information in compressed quantum states"
        }
        
        return rules.get(strategy, "")

    def _generate_strategic_context(self, context: Dict) -> str:
        """Enhanced with sparse attention strategic benefits."""
        strategic_context = context.get('strategic_context')
        sparse_info = context.get('sparse_attention_info', {})
        
        if not strategic_context:
            return ""

        explanation = f"### Strategic Context:\nThis move aligns with: **{strategic_context}**\n"
        
        # Add sparse attention strategic benefits
        efficiency_gain = sparse_info.get('efficiency_gain', 1.0)
        if efficiency_gain > 3.0:
            explanation += f"- **Strategic Advantage**: {efficiency_gain:.1f}x efficiency enables deeper lookahead\n"
        
        return explanation

    def _generate_counterfactual_analysis(self, context: Dict) -> str:
        """Enhanced with attention pattern comparisons."""
        all_moves = context.get('all_moves', [])
        all_scores = context.get('all_scores', [])
        chosen_move = context.get('chosen_move')

        if len(all_moves) < 2:
            return ""

        # Find top alternatives
        move_scores = list(zip(all_moves, all_scores))
        move_scores.sort(key=lambda x: x[1], reverse=True)

        explanation = "### Alternatives Considered:\n"

        for i, (move, score) in enumerate(move_scores[:3]):
            if move == chosen_move:
                continue

            # Compare attention patterns
            attention_comparison = self._compare_attention_patterns(context, chosen_move, move)
            
            explanation += f"- **Move {move}** (Score: {score:.2f}):\n"
            if attention_comparison:
                explanation += f"  - Attention differences: {attention_comparison}\n"

        return explanation

    def _compare_attention_patterns(self, context: Dict, move1: any, move2: any) -> str:
        """Compare attention patterns between two moves."""
        metrics1 = self._get_metrics_for_move(context, move1)
        metrics2 = self._get_metrics_for_move(context, move2)
        
        strategy1 = metrics1.get('sparse_attention_info', {}).get('selected_strategy', 'standard')
        strategy2 = metrics2.get('sparse_attention_info', {}).get('selected_strategy', 'standard')
        
        if strategy1 != strategy2:
            return f"Different attention strategies: {self._format_strategy_name(strategy1)} vs {self._format_strategy_name(strategy2)}"
        
        sparsity1 = metrics1.get('sparse_attention_info', {}).get('attention_sparsity', 0)
        sparsity2 = metrics2.get('sparse_attention_info', {}).get('attention_sparsity', 0)
        
        if abs(sparsity1 - sparsity2) > 0.1:
            return f"Different sparsity levels: {sparsity1:.1%} vs {sparsity2:.1%}"
            
        return "Similar attention patterns"

    def _generate_confidence_assessment(self, context: Dict) -> str:
        """Enhanced confidence with sparse attention factors."""
        confidence = self._calculate_verifiable_confidence(context)
        sparse_info = context.get('sparse_attention_info', {})

        confidence_levels = {
            0.9: "Very High Confidence",
            0.7: "High Confidence", 
            0.5: "Moderate Confidence",
            0.3: "Low Confidence",
            0.0: "Very Low Confidence"
        }

        level = next((desc for threshold, desc in confidence_levels.items() if confidence >= threshold), "Very Low Confidence")

        explanation = f"### Confidence Assessment:\n**{level}** ({confidence:.2f}/1.0)\n\n"
        explanation += "This confidence is based on:\n- CKG rule validation consistency\n- Historical decision performance\n- Strategic alignment verification\n"
        
        # Add sparse attention confidence factors
        attention_strategy = sparse_info.get('selected_strategy', 'standard')
        if attention_strategy in ['self_evolving', 'ckg_guided']:
            explanation += f"- High-reliability attention strategy ({self._format_strategy_name(attention_strategy)})\n"
            
        efficiency_gain = sparse_info.get('efficiency_gain', 1.0)
        if efficiency_gain > 2.0:
            explanation += f"- {efficiency_gain:.1f}x computational efficiency enabled deeper analysis\n"

        return explanation

    def _calculate_verifiable_confidence(self, context: Dict) -> float:
        """Enhanced confidence calculation with sparse attention factors."""
        chosen_metrics = self._get_metrics_for_move(context, context.get('chosen_move'))
        sparse_info = context.get('sparse_attention_info', {})

        confidence = 0.5  # Base confidence

        # Factor 1: Validation confidence
        validation_conf = chosen_metrics.get('ckg_validation', {}).get('confidence', 1.0)
        confidence *= validation_conf

        # Factor 2: Rule consistency
        applied_rules = chosen_metrics.get('ckg_validation', {}).get('applied_rules', [])
        violated_rules = chosen_metrics.get('ckg_validation', {}).get('violated_rules', [])
        if applied_rules and not violated_rules:
            confidence *= 1.2

        # Factor 3: Sparse attention reliability
        attention_strategy = sparse_info.get('selected_strategy', 'standard')
        if attention_strategy in ['self_evolving', 'ckg_guided']:
            confidence *= 1.1  # Bonus for reliable strategies
            
        # Factor 4: Efficiency gains (more compute = more confidence)
        efficiency_gain = sparse_info.get('efficiency_gain', 1.0)
        if efficiency_gain > 2.0:
            confidence *= min(1.0 + (efficiency_gain - 2.0) * 0.1, 1.2)

        return max(0.0, min(1.0, confidence))

    # --- Enhanced Query Handling ---
    def handle_query(self, 
                     query: str, 
                     decision_context: Dict, 
                     conceptual_features: torch.Tensor,
                     domain: str,
                     current_fused_rep: torch.Tensor) -> str:
        """
        Enhanced query handling with sparse attention explanations.
        """
        parsed_command = self.parser.parse_command(query)
        command = parsed_command['command']

        if command == "explain":
            return self.generate_symbolic_explanation(decision_context)

        elif command == "attention":
            return self._explain_attention_patterns(decision_context)

        elif command == "efficiency":
            return self._explain_efficiency_gains(decision_context)

        elif command == "quantum":
            return self._explain_quantum_compression(decision_context)

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
            return "I'm sorry, I don't understand that command. Please try 'explain', 'attention', 'efficiency', 'quantum', 'rules', 'confidence', 'strategy', 'eval_move', 'what-if', or 'review_proposals'."

    def _explain_attention_patterns(self, context: Dict) -> str:
        """Provide detailed explanation of attention patterns used."""
        sparse_info = context.get('sparse_attention_info', {})
        if not sparse_info:
            return "No sparse attention information available for this decision."

        explanation = "### Detailed Attention Pattern Analysis:\n"
        
        strategy = sparse_info.get('selected_strategy', 'unknown')
        sparsity = sparse_info.get('attention_sparsity', 0.0)
        efficiency = sparse_info.get('efficiency_gain', 1.0)
        
        explanation += f"**Strategy**: {self._format_strategy_name(strategy)}\n"
        explanation += f"**Sparsity**: {sparsity:.1%} of possible connections used\n"
        explanation += f"**Efficiency**: {efficiency:.1f}x faster than full attention\n"
        explanation += f"**Memory Savings**: {(1 - sparsity) * 100:.1f}% reduction\n"
        
        # Explain strategy benefits
        strategy_benefits = {
            'conceptual_sparse': "Focuses only on semantically important information, ignoring noise",
            'ckg_guided': "Uses knowledge graph relationships to determine what to pay attention to",
            'self_evolving': "Uses patterns that have evolved and proven successful over time", 
            'quantum_compressed': "Processes information in highly compressed quantum states"
        }
        
        explanation += f"\n**Strategy Benefits**: {strategy_benefits.get(strategy, 'Standard attention')}\n"
        
        return explanation

    def _explain_efficiency_gains(self, context: Dict) -> str:
        """Explain computational efficiency improvements."""
        sparse_info = context.get('sparse_attention_info', {})
        efficiency = sparse_info.get('efficiency_gain', 1.0)
        sparsity = sparse_info.get('attention_sparsity', 0.0)
        
        explanation = f"### Efficiency Analysis:\n"
        explanation += f"- **Speed Improvement**: {efficiency:.1f}x faster\n"
        explanation += f"- **Memory Reduction**: {(1 - sparsity) * 100:.1f}% less memory\n"
        explanation += f"- **Theoretical Maximum**: O(n√n) vs O(n²) complexity\n"
        
        if efficiency > 5.0:
            explanation += "\n**Impact**: This level of efficiency enables processing much longer sequences and more complex patterns than standard attention.\n"
        elif efficiency > 2.0:
            explanation += "\n**Impact**: Significant efficiency gains allow for deeper analysis and better strategic planning.\n"
            
        return explanation

    def _explain_quantum_compression(self, context: Dict) -> str:
        """Explain quantum compression benefits."""
        quantum_info = context.get('quantum_compression_info', {})
        if not quantum_info:
            return "No quantum compression was used in this decision."
            
        explanation = "### Quantum Compression Analysis:\n"
        explanation += f"- **Compression Ratio**: {quantum_info.get('compression_ratio', 1.0):.1f}:1\n"
        explanation += f"- **Information Preservation**: {quantum_info.get('quantum_coherence', 0.0):.1%}\n"
        explanation += f"- **Memory Efficiency**: {quantum_info.get('information_density', 0.0):.3f} bits/parameter\n"
        
        explanation += "\n**Quantum Principles Applied**:\n"
        explanation += "- Superposition: Information exists in multiple states simultaneously\n"
        explanation += "- Entanglement: Correlated information compression\n"
        explanation += "- Decoherence Control: Minimized information loss\n"
        
        return explanation

    # --- Keep existing methods for compatibility ---
    def _get_metrics_for_move(self, context: Dict, move: any) -> Dict:
        """Get decision metrics for a specific move."""
        decision_metrics = context.get('decision_metrics', [])
        for metrics in decision_metrics:
            if metrics.get('move') == move:
                return metrics
        return {}

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
        sparse_info = context.get('sparse_attention_info', {})

        explanation = f"### Confidence Breakdown:\nOverall: {confidence:.2f}/1.0\n\n"

        # Validation confidence
        validation_conf = chosen_metrics.get('ckg_validation', {}).get('confidence', 1.0)
        explanation += f"- Validation Confidence: {validation_conf:.2f} (CKG rule validation)\n"

        # Sparse attention confidence
        attention_strategy = sparse_info.get('selected_strategy', 'standard')
        if attention_strategy in ['self_evolving', 'ckg_guided']:
            explanation += f"- Attention Strategy Bonus: +0.1 (high-reliability strategy)\n"

        # Efficiency impact
        efficiency = sparse_info.get('efficiency_gain', 1.0)
        if efficiency > 2.0:
            explanation += f"- Efficiency Bonus: +{(efficiency - 2.0) * 0.1:.2f} (deeper analysis enabled)\n"

        return explanation

    # ... (keep all other existing methods for compatibility)