"""
Counterfactual Reasoning Verification Tests
===========================================
Tests to verify the system's ability to reason about alternative actions
and outcomes (what-if scenarios).
"""

import unittest
import torch
import numpy as np
from src.asreh_algorithm import ASREHAlgorithm
from src.conceptual_knowledge_graph.ckg import ConceptualKnowledgeGraph
from src.models.explainability_module import ExplainabilityModule
from src.models.sswm import SSWM
from src.models.arlc_controller import ARLCController
from src.models.asreh_model import ASREHModel

class TestCounterfactualReasoning(unittest.TestCase):
    """Test suite for counterfactual reasoning capabilities."""
    
    def setUp(self):
        """Set up test fixtures with mock components."""
        self.ckg = ConceptualKnowledgeGraph()
        self._setup_test_knowledge_base()
        
        # Mock components (in practice, these would be properly initialized)
        self.model = ASREHModel(ckg=self.ckg)
        self.sswm = SSWM()
        self.arlc = ARLCController(strategic_planner=None, sswm=self.sswm, ckg=self.ckg)
        self.em = ExplainabilityModule(model=self.model, sswm=self.sswm, ckg=self.ckg)
        
    def _setup_test_knowledge_base(self):
        """Populate CKG with test rules for counterfactual reasoning."""
        # Add game concepts
        self.ckg.add_node("high_score_move", {
            "type": "strategic_concept",
            "description": "A move that maximizes immediate points"
        })
        
        self.ckg.add_node("strategic_move", {
            "type": "strategic_concept", 
            "description": "A move that sets up future advantages"
        })
        
        self.ckg.add_node("exploratory_move", {
            "type": "strategic_concept",
            "description": "A move that explores new possibilities"
        })
        
        # Add causal relationships
        self.ckg.add_edge("high_score_move", "immediate_reward", "PROVIDES", 
                         {"value": 10.0, "certainty": 0.9})
        self.ckg.add_edge("strategic_move", "future_advantage", "PROVIDES",
                         {"value": 15.0, "certainty": 0.7})
        self.ckg.add_edge("exploratory_move", "knowledge_gain", "PROVIDES",
                         {"value": 8.0, "certainty": 0.6})

    def test_alternative_action_evaluation(self):
        """Test that the system can evaluate and compare alternative actions."""
        # Create mock decision context with multiple action options
        decision_context = {
            'chosen_action': 'move_3',
            'chosen_score': 12.5,
            'all_actions': ['move_1', 'move_2', 'move_3', 'move_4'],
            'all_scores': [8.2, 10.1, 12.5, 9.8],
            'forecasts': {
                'move_1': {'ckg_validation': {'applied_rules': ['rule_conservative_play']}},
                'move_2': {'ckg_validation': {'applied_rules': ['rule_balanced_approach']}},
                'move_3': {'ckg_validation': {'applied_rules': ['rule_aggressive_strategy']}},
                'move_4': {'ckg_validation': {'applied_rules': ['rule_exploratory_play']}}
            },
            'domain': 'tetris'
        }
        
        # Generate explanation with counterfactual analysis
        explanation = self.em.generate_explanation(decision_context)
        
        # Verify counterfactual reasoning is present
        self.assertIn("alternative", explanation.lower(),
                     "Explanation should discuss alternative actions")
        self.assertIn("move_2", explanation or "move_4" in explanation,
                     "Should mention specific alternative moves")
        
    def test_what_if_scenario_generation(self):
        """Test the system's ability to generate what-if scenarios."""
        # Mock SSWM simulation capability
        mock_current_state = torch.randn(1, 128)  # Mock fused representation
        
        # Test multiple hypothetical moves
        hypothetical_moves = ['move_alternative_1', 'move_alternative_2']
        simulation_results = {}
        
        for move in hypothetical_moves:
            # Simulate what would happen with this alternative move
            simulated_outcome = self.sswm.simulate_what_if_scenario(
                mock_current_state, move, num_steps=3
            )
            simulation_results[move] = simulated_outcome
            
            # Verify simulation provides meaningful data
            self.assertIn('predicted_reward', simulated_outcome,
                         "Simulation should provide reward prediction")
            self.assertIn('conceptual_changes', simulated_outcome,
                         "Simulation should show conceptual impact")
        
        # Verify we got results for all hypothetical moves
        self.assertEqual(len(simulation_results), len(hypothetical_moves),
                        "Should simulate all hypothetical moves")
        
    def test_counterfactual_explanation_quality(self):
        """Test that counterfactual explanations are meaningful and accurate."""
        decision_context = {
            'chosen_action': 'optimal_move',
            'chosen_score': 15.0,
            'all_actions': ['optimal_move', 'good_alternative', 'poor_choice'],
            'all_scores': [15.0, 12.0, 5.0],
            'forecasts': {
                'optimal_move': {
                    'ckg_validation': {
                        'applied_rules': ['rule_optimal_placement'],
                        'confidence': 0.95
                    }
                },
                'good_alternative': {
                    'ckg_validation': {
                        'applied_rules': ['rule_good_but_suboptimal'],
                        'confidence': 0.85
                    }
                },
                'poor_choice': {
                    'ckg_validation': {
                        'violated_rules': ['rule_avoid_bad_placement'],
                        'confidence': 0.3
                    }
                }
            },
            'domain': 'tetris'
        }
        
        explanation = self.em.generate_explanation(decision_context)
        
        # Check explanation quality metrics
        self._verify_explanation_quality(explanation, decision_context)
        
    def _verify_explanation_quality(self, explanation: str, context: Dict):
        """Verify that counterfactual explanation meets quality standards."""
        # Should mention the chosen action
        self.assertIn(context['chosen_action'].lower(), explanation.lower(),
                     "Should mention the chosen action")
        
        # Should discuss at least one alternative
        alternatives_mentioned = any(
            alt_action.lower() in explanation.lower() 
            for alt_action in context['all_actions'] 
            if alt_action != context['chosen_action']
        )
        self.assertTrue(alternatives_mentioned,
                       "Should discuss alternative actions")
        
        # Should provide reasoning for rejection of alternatives
        self.assertTrue(
            any(keyword in explanation.lower() 
                for keyword in ['because', 'reason', 'why', 'due to']),
            "Should provide causal reasoning"
        )
        
    def test_strategic_tradeoff_analysis(self):
        """Test that the system can analyze strategic tradeoffs between actions."""
        # Create scenario with different strategic profiles
        action_profiles = {
            'aggressive_move': {
                'immediate_gain': 12.0,
                'future_risk': 0.6,
                'conceptual_tags': ['high_reward', 'high_risk']
            },
            'conservative_move': {
                'immediate_gain': 8.0, 
                'future_risk': 0.2,
                'conceptual_tags': ['low_reward', 'low_risk']
            },
            'balanced_move': {
                'immediate_gain': 10.0,
                'future_risk': 0.4,
                'conceptual_tags': ['moderate_reward', 'moderate_risk']
            }
        }
        
        # Analyze tradeoffs (this would be done by ARLC in practice)
        tradeoff_analysis = self._analyze_strategic_tradeoffs(action_profiles)
        
        # Verify analysis covers key aspects
        self.assertIn('risk_reward_ratio', tradeoff_analysis,
                     "Should analyze risk-reward tradeoffs")
        self.assertIn('strategic_alignment', tradeoff_analysis,
                     "Should assess strategic alignment")
        self.assertIn('recommendation', tradeoff_analysis,
                     "Should provide a recommendation")
        
    def _analyze_strategic_tradeoffs(self, action_profiles: Dict) -> Dict:
        """Mock strategic tradeoff analysis."""
        analysis = {}
        
        for action, profile in action_profiles.items():
            # Calculate risk-reward ratio
            risk_reward = profile['immediate_gain'] / (profile['future_risk'] + 0.001)
            analysis[action] = {
                'risk_reward_ratio': risk_reward,
                'strategic_value': self._assess_strategic_value(profile['conceptual_tags'])
            }
        
        # Generate overall recommendation
        best_action = max(action_profiles.keys(), 
                         key=lambda a: analysis[a]['risk_reward_ratio'] * analysis[a]['strategic_value'])
        
        analysis['recommendation'] = {
            'best_action': best_action,
            'reasoning': f"Optimal balance of risk-reward ratio and strategic value"
        }
        
        return analysis
    
    def _assess_strategic_value(self, conceptual_tags: List[str]) -> float:
        """Mock strategic value assessment based on conceptual tags."""
        strategic_weights = {
            'high_reward': 1.2,
            'low_risk': 1.1, 
            'moderate_reward': 1.0,
            'moderate_risk': 0.9,
            'high_risk': 0.7
        }
        
        return sum(strategic_weights.get(tag, 0.5) for tag in conceptual_tags) / len(conceptual_tags)

if __name__ == '__main__':
    unittest.main()
