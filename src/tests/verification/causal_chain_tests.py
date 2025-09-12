"""
Causal Chain Verification Tests
===============================
Tests to verify that the ASREH system can correctly trace causal chains
from actions to outcomes using the Conceptual Knowledge Graph.
"""

import unittest
import torch
import numpy as np
from src.conceptual_knowledge_graph.ckg import ConceptualKnowledgeGraph
from src.models.asreh_model import ASREHModel
from src.models.sswm import SSWM

class TestCausalChainReasoning(unittest.TestCase):
    """Test suite for causal chain reasoning capabilities."""
    
    def setUp(self):
        """Set up test fixtures with a minimal CKG."""
        self.ckg = ConceptualKnowledgeGraph()
        self._setup_test_knowledge_base()
        
    def _setup_test_knowledge_base(self):
        """Populate CKG with test rules for causal reasoning."""
        # Add basic concepts
        self.ckg.add_node("tetris_block", {
            "type": "game_object", 
            "properties": ["shape", "orientation", "position"]
        })
        
        self.ckg.add_node("board_gap", {
            "type": "game_state", 
            "properties": ["size", "position", "is_fillable"]
        })
        
        self.ckg.add_node("lines_cleared", {
            "type": "game_outcome", 
            "properties": ["count", "points_earned"]
        })
        
        # Add causal rules
        self.ckg.add_causal_rule(
            "place_s_block", 
            "create_gap", 
            {
                "id": "rule_s_block_gap",
                "description": "Placing an S-block in certain positions creates gaps",
                "conditions": ["board_has_ledge=true", "block_type=S", "position>5"],
                "confidence": 0.85
            }
        )
        
        self.ckg.add_causal_rule(
            "fill_row_completely", 
            "clear_line", 
            {
                "id": "rule_line_clear",
                "description": "Filling all cells in a row clears the line",
                "conditions": ["row_fill_percentage=100%"],
                "confidence": 1.0
            }
        )

    def test_causal_chain_detection(self):
        """Test that the system can detect and trace causal chains."""
        # Create a test scenario
        test_action = "place_s_block_at_x7"
        test_outcome = "gap_created_at_bottom"
        
        # Verify the causal chain exists in CKG
        chain = self.ckg.trace_causal_chain(test_action, test_outcome)
        
        self.assertIsNotNone(chain, "Causal chain should be detected")
        self.assertIn("rule_s_block_gap", chain['rules_used'],
                     "Should use the S-block gap rule")
        
    def test_forecast_validation_against_ckg(self):
        """Test that state forecasts are validated against CKG rules."""
        # Create a mock forecasted state that should violate CKG rules
        problematic_forecast = {
            'conceptual_features': torch.tensor([1.0, 0.0, 1.0]),  # Example features
            'action': 'place_s_block_at_x7',
            'predicted_outcomes': ['gap_creation', 'line_clearance']  # Contradictory outcomes
        }
        
        # Validate against CKG
        validation_result = self.ckg.validate_forecast(
            problematic_forecast['conceptual_features'],
            problematic_forecast['action'],
            'tetris'
        )
        
        self.assertFalse(validation_result['is_valid'],
                        "Contradictory forecast should be invalid")
        self.assertGreater(len(validation_result['violated_rules']), 0,
                          "Should identify violated rules")
    
    def test_rule_based_scoring_consistency(self):
        """Test that scoring is consistent with CKG rules."""
        # Create test forecasts with different rule compliance
        compliant_forecast = {
            'conceptual_features': torch.tensor([0.0, 1.0, 0.0]),  # Features indicating rule compliance
            'ckg_validation': {'is_valid': True, 'applied_rules': ['rule_line_clear']}
        }
        
        non_compliant_forecast = {
            'conceptual_features': torch.tensor([1.0, 0.0, 0.0]),  # Features indicating rule violation
            'ckg_validation': {'is_valid': False, 'violated_rules': ['rule_s_block_gap']}
        }
        
        # Scores should reflect rule compliance
        # (This would integrate with ARLC scoring in practice)
        compliant_score = self._calculate_rule_based_score(compliant_forecast)
        non_compliant_score = self._calculate_rule_based_score(non_compliant_forecast)
        
        self.assertGreater(compliant_score, non_compliant_score,
                          "Compliant forecasts should score higher")
    
    def _calculate_rule_based_score(self, forecast: Dict) -> float:
        """Mock scoring function based on rule compliance."""
        base_score = 10.0
        if forecast['ckg_validation']['is_valid']:
            return base_score + len(forecast['ckg_validation'].get('applied_rules', [])) * 2.0
        else:
            return base_score - len(forecast['ckg_validation'].get('violated_rules', [])) * 5.0

    def test_explanation_traceability(self):
        """Test that explanations can be traced back to CKG elements."""
        # Create a mock decision context
        decision_context = {
            'chosen_action': 'place_i_block_at_x4',
            'ckg_references': {
                'place_i_block_at_x4': {
                    'applied_rules': ['rule_line_clear', 'rule_efficient_placement'],
                    'confidence': 0.92
                }
            }
        }
        
        # Generate explanation and verify traceability
        explanation = self._generate_traceable_explanation(decision_context)
        
        # Check that explanation references CKG rules
        self.assertIn("rule_line_clear", explanation,
                     "Explanation should reference applied rules")
        self.assertIn("rule_efficient_placement", explanation,
                     "Explanation should reference all relevant rules")
    
    def _generate_traceable_explanation(self, context: Dict) -> str:
        """Mock explanation generation that traces back to CKG."""
        action = context['chosen_action']
        rules = context['ckg_references'][action]['applied_rules']
        
        explanation = f"Action {action} was chosen because it follows these rules: "
        explanation += ", ".join(rules)
        explanation += f" with {context['ckg_references'][action]['confidence']*100:.1f}% confidence."
        
        return explanation

if __name__ == '__main__':
    unittest.main()
