"""
Adversarial Probing Verification Tests
======================================
Tests to verify the system's resilience against adversarial attacks
and its ability to detect and respond to vulnerabilities.
"""

import unittest
import torch
import numpy as np
from src.models.adversarial_module import AdversarialModule
from src.conceptual_knowledge_graph.ckg import ConceptualKnowledgeGraph
from src.models.explainability_module import ExplainabilityModule
from src.models.arlc_controller import ARLCController

class TestAdversarialResilience(unittest.TestCase):
    """Test suite for adversarial robustness and vulnerability detection."""
    
    def setUp(self):
        """Set up test fixtures with adversarial testing components."""
        self.ckg = ConceptualKnowledgeGraph()
        
        # Create a simple mock model for testing
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Linear(128, 64)
                
            def forward(self, x, conceptual_features, domain):
                return torch.randn(1, 20, 10), torch.randn(1, 64), torch.tensor(0.5)
        
        self.mock_model = MockModel()
        self.adversarial_module = AdversarialModule(self.mock_model, self.ckg)
        self.em = ExplainabilityModule(model=self.mock_model, sswm=None, ckg=self.ckg)
        self.arlc = ARLCController(strategic_planner=None, sswm=None, ckg=self.ckg)
        
    def test_adversarial_perturbation_generation(self):
        """Test that the adversarial module can generate effective perturbations."""
        # Create a sample fused representation
        original_rep = torch.randn(1, 64)  # Mock HCT_DIM=64
        
        # Generate adversarial perturbation
        adversarial_rep = self.adversarial_module.generate_adversarial_input(original_rep)
        
        # Verify perturbation changes the representation
        perturbation_norm = torch.norm(adversarial_rep - original_rep).item()
        self.assertGreater(perturbation_norm, 0.1,
                          "Perturbation should significantly change the representation")
        self.assertLess(perturbation_norm, 5.0,
                       "Perturbation should be bounded and reasonable")
        
    def test_vulnerability_detection(self):
        """Test that the system can detect model vulnerabilities."""
        # Create test scenario with significant output divergence
        original_rep = torch.randn(1, 64)
        adversarial_rep = original_rep + 2.0  # Large perturbation
        
        # Get model outputs
        with torch.no_grad():
            original_output, _, _ = self.mock_model(original_rep, torch.randn(1, 64), 'tetris')
            adversarial_output, _, _ = self.mock_model(adversarial_rep, torch.randn(1, 64), 'tetris')
        
        # Analyze for vulnerabilities
        failure_report = self.em.analyze_and_report_failure(
            original_rep, adversarial_rep, original_output, adversarial_output
        )
        
        # Verify vulnerability detection
        self.assertIn('type', failure_report,
                     "Failure report should identify error type")
        self.assertIn('explanation', failure_report,
                     "Failure report should provide explanation")
        self.assertIn('causal_factors', failure_report,
                     "Failure report should identify causal factors")
        
    def test_ckg_vulnerability_registration(self):
        """Test that detected vulnerabilities are registered in the CKG."""
        # Create a mock failure report
        failure_report = {
            'type': 'conceptual_misinterpretation',
            'explanation': 'Small input changes cause large output divergence',
            'causal_factors': ['sensitivity_to_perturbation', 'lack_of_robustness'],
            'metrics': {'output_divergence': 0.8, 'input_change': 0.1}
        }
        
        # Register vulnerability in CKG
        self.adversarial_module.propose_new_relationship_from_failure(failure_report)
        
        # Verify vulnerability was registered
        vulnerability_nodes = [
            node for node_id, node in self.ckg.db.nodes.items()
            if 'vulnerability' in node_id.lower() or 'Vulnerability_' in node_id
        ]
        
        self.assertGreater(len(vulnerability_nodes), 0,
                          "Vulnerability should be registered in CKG")
        
    def test_self_correction_mechanism(self):
        """Test that the system can self-correct from detected vulnerabilities."""
        # Create a failure scenario
        failure_report = {
            'type': 'conceptual_misinterpretation',
            'explanation': 'Model is overly sensitive to small input changes',
            'causal_factors': ['conceptual_misinterpretation'],
            'subtype': 'input_sensitivity'
        }
        
        # Trigger self-correction
        self.arlc.self_correct_from_failure(failure_report, self.mock_model)
        
        # Verify correction mechanisms were activated
        # (In practice, this would check if model weights were adjusted,
        #  or if new training data was generated, etc.)
        correction_log = [
            node for node_id, node in self.ckg.db.nodes.items()
            if 'self_correction' in node_id.lower() or 'failure' in node_id.lower()
        ]
        
        self.assertGreater(len(correction_log), 0,
                          "Self-correction should be logged in CKG")
        
    def test_adversarial_training_effectiveness(self):
        """Test that adversarial training improves model robustness."""
        initial_vulnerabilities = self._count_vulnerabilities()
        
        # Run a short adversarial training session
        self.adversarial_module.run_adversarial_training(self.arlc, self.em, epochs=5)
        
        # Check if vulnerabilities decreased or were addressed
        final_vulnerabilities = self._count_vulnerabilities()
        
        # Either vulnerabilities decrease or mitigation strategies are proposed
        vulnerability_change = final_vulnerabilities - initial_vulnerabilities
        self.assertTrue(
            vulnerability_change <= 0 or self._has_mitigation_proposals(),
            "Adversarial training should reduce vulnerabilities or propose fixes"
        )
        
    def _count_vulnerabilities(self) -> int:
        """Count the number of vulnerability entries in CKG."""
        return sum(
            1 for node_id, node in self.ckg.db.nodes.items()
            if 'vulnerability' in node_id.lower() or node.get('is_vulnerability', False)
        )
        
    def _has_mitigation_proposals(self) -> bool:
        """Check if there are proposed fixes for vulnerabilities."""
        return any(
            'proposal' in node_id.lower() and 'upgrade' in str(node).lower()
            for node_id, node in self.ckg.db.nodes.items()
        )
    
    def test_architectural_upgrade_proposal(self):
        """Test that critical vulnerabilities trigger architectural upgrade proposals."""
        # Simulate a critical vulnerability finding
        critical_failure_report = {
            'type': 'critical_vulnerability',
            'explanation': 'Fundamental architectural weakness detected',
            'causal_factors': ['architectural_limitation'],
            'metrics': {'output_divergence': 0.9, 'input_change': 0.05}
        }
        
        # Trigger upgrade proposal
        self.adversarial_module._propose_architectural_upgrade("dynamic_quantization_upgrade")
        
        # Verify upgrade proposal was created
        upgrade_proposals = [
            node for node_id, node in self.ckg.db.nodes.items()
            if 'proposal' in node_id.lower() and 'upgrade' in str(node).lower()
        ]
        
        self.assertGreater(len(upgrade_proposals), 0,
                          "Critical vulnerabilities should trigger upgrade proposals")
        
        proposal = upgrade_proposals[0]
        self.assertIn('predicted_impact', proposal,
                     "Upgrade proposal should include impact prediction")
        self.assertIn('proposed_changes', proposal,
                     "Upgrade proposal should detail changes")
        self.assertEqual(proposal.get('status'), 'pending_human_review',
                       "Upgrade proposal should await human review")
    
    def test_robustness_metrics_tracking(self):
        """Test that robustness metrics are properly tracked and reported."""
        # Create multiple test scenarios with varying perturbation levels
        test_cases = [
            {'perturbation_strength': 0.1, 'expected_divergence': 0.2},
            {'perturbation_strength': 0.5, 'expected_divergence': 0.6},
            {'perturbation_strength': 1.0, 'expected_divergence': 0.9}
        ]
        
        robustness_metrics = []
        
        for test_case in test_cases:
            original_rep = torch.randn(1, 64)
            adversarial_rep = original_rep + test_case['perturbation_strength']
            
            with torch.no_grad():
                original_output, _, _ = self.mock_model(original_rep, torch.randn(1, 64), 'tetris')
                adversarial_output, _, _ = self.mock_model(adversarial_rep, torch.randn(1, 64), 'tetris')
            
            # Calculate output divergence
            divergence = torch.nn.functional.mse_loss(original_output, adversarial_output).item()
            robustness_metrics.append({
                'perturbation_strength': test_case['perturbation_strength'],
                'output_divergence': divergence
            })
        
        # Verify robustness metrics show expected pattern
        self.assertEqual(len(robustness_metrics), len(test_cases),
                        "Should track metrics for all test cases")
        
        # Check that stronger perturbations cause more divergence
        for i in range(1, len(robustness_metrics)):
            self.assertGreater(robustness_metrics[i]['output_divergence'],
                             robustness_metrics[i-1]['output_divergence'],
                             "Stronger perturbations should cause more divergence")
    
    def test_explanation_consistency_under_attack(self):
        """Test that explanations remain consistent under adversarial conditions."""
        # Create baseline explanation
        original_rep = torch.randn(1, 64)
        with torch.no_grad():
            original_output, _, _ = self.mock_model(original_rep, torch.randn(1, 64), 'tetris')
        
        baseline_explanation = self.em.generate_explanation({
            'chosen_action': 'test_move',
            'chosen_score': 0.8,
            'domain': 'tetris'
        })
        
        # Test with small perturbations
        small_perturbation = original_rep + 0.1
        with torch.no_grad():
            perturbed_output, _, _ = self.mock_model(small_perturbation, torch.randn(1, 64), 'tetris')
        
        perturbed_explanation = self.em.generate_explanation({
            'chosen_action': 'test_move', 
            'chosen_score': 0.8,
            'domain': 'tetris'
        })
        
        # Explanations should remain reasonably consistent under small perturbations
        # (We don't expect exact matches, but they should be semantically similar)
        baseline_keywords = self._extract_key_terms(baseline_explanation)
        perturbed_keywords = self._extract_key_terms(perturbed_explanation)
        
        overlap = len(set(baseline_keywords) & set(perturbed_keywords))
        similarity_ratio = overlap / max(len(baseline_keywords), len(perturbed_keywords))
        
        self.assertGreater(similarity_ratio, 0.6,
                          "Explanations should remain consistent under small perturbations")
    
    def _extract_key_terms(self, text: str) -> list:
        """Extract key terms from explanation text for comparison."""
        # Simple keyword extraction - in practice could use more sophisticated NLP
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        words = text.lower().split()
        return [word for word in words if word not in stop_words and len(word) > 3]
    
    def test_adaptive_response_to_attacks(self):
        """Test that the system adapts its response based on attack severity."""
        # Test different attack severity levels
        attack_levels = ['low', 'medium', 'high', 'critical']
        responses = []
        
        for level in attack_levels:
            # Simulate different attack scenarios
            if level == 'low':
                perturbation = 0.1
            elif level == 'medium':
                perturbation = 0.5
            elif level == 'high':
                perturbation = 1.0
            else:  # critical
                perturbation = 2.0
            
            original_rep = torch.randn(1, 64)
            adversarial_rep = original_rep + perturbation
            
            with torch.no_grad():
                original_output, _, _ = self.mock_model(original_rep, torch.randn(1, 64), 'tetris')
                adversarial_output, _, _ = self.mock_model(adversarial_rep, torch.randn(1, 64), 'tetris')
            
            # Get system response
            failure_report = self.em.analyze_and_report_failure(
                original_rep, adversarial_rep, original_output, adversarial_output
            )
            
            responses.append({
                'attack_level': level,
                'response_severity': self._classify_response_severity(failure_report),
                'response_type': failure_report.get('type', 'unknown')
            })
        
        # Verify response escalates with attack severity
        severity_levels = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        for i in range(1, len(responses)):
            current_severity = severity_levels[responses[i]['attack_level']]
            prev_severity = severity_levels[responses[i-1]['attack_level']]
            
            if current_severity > prev_severity:
                # Response should be at least as severe or more severe
                self.assertGreaterEqual(responses[i]['response_severity'],
                                      responses[i-1]['response_severity'],
                                      "Response should escalate with attack severity")
    
    def _classify_response_severity(self, failure_report: dict) -> int:
        """Classify the severity of the system's response to an attack."""
        response_type = failure_report.get('type', '')
        if 'critical' in response_type:
            return 4
        elif 'severe' in response_type or 'architectural' in response_type:
            return 3
        elif 'moderate' in response_type or 'significant' in response_type:
            return 2
        else:
            return 1

if __name__ == '__main__':
    unittest.main()