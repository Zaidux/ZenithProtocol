# /src/models/adversarial_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Dict, List, Tuple, Any
from ..utils.config import Config
from ..conceptual_knowledge_graph.ckg import ConceptualKnowledgeGraph
from .architectural_reconfiguration_layer import ArchitecturalReconfigurationLayer # New Import

class AdversarialModule(nn.Module):
    def __init__(self, model: nn.Module, ckg: ConceptualKnowledgeGraph):
        super(AdversarialModule, self).__init__()
        self.model = model
        self.ckg = ckg
        self.config = Config()
        self.generator = nn.Sequential(
            nn.Linear(self.config.HCT_DIM, self.config.HCT_DIM * 2),
            nn.ReLU(),
            nn.Linear(self.config.HCT_DIM * 2, self.config.HCT_DIM),
        ).to(self.config.DEVICE)
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.config.ADVERSARIAL_LR)
        self.adversarial_loss_fn = nn.MSELoss()
        self.arch_layer = ArchitecturalReconfigurationLayer(model, ckg) # New instance

    def generate_adversarial_input(self, fused_representation: torch.Tensor) -> torch.Tensor:
        perturbation = self.generator(fused_representation)
        adversarial_input = fused_representation + perturbation
        return adversarial_input

    def run_adversarial_training(self, arlc, em):
        print("\n[Adversary] Starting adversarial training loop...")
        self.generator.train()
        self.model.eval()

        for epoch in range(self.config.ADVERSARIAL_EPOCHS):
            random_fused_rep = torch.randn(1, self.config.HCT_DIM).to(self.config.DEVICE)
            adversarial_rep = self.generate_adversarial_input(random_fused_rep)
            original_output, _, _ = self.model.forward(random_fused_rep, torch.randn(1, 64), 'tetris')
            adversarial_output, _, _ = self.model.forward(adversarial_rep, torch.randn(1, 64), 'tetris')
            adversarial_loss = -self.adversarial_loss_fn(original_output, adversarial_output)
            self.optimizer.zero_grad()
            adversarial_loss.backward()
            self.optimizer.step()

            if (epoch + 1) % self.config.ADVERSARIAL_LOG_INTERVAL == 0:
                print(f"[Adversary] Epoch {epoch+1}/{self.config.ADVERSARIAL_EPOCHS}, Adversarial Loss: {adversarial_loss.item():.4f}")
                if adversarial_loss.item() < -0.5:
                    print("\n[Adversary] Major weakness found! Generating failure report...")
                    failure_report = em.analyze_and_report_failure(
                        original_input=random_fused_rep,
                        adversarial_input=adversarial_rep,
                        original_output=original_output,
                        adversarial_output=adversarial_output
                    )
                    self.propose_new_relationship_from_failure(failure_report)
                    
                    # New: Propose architectural upgrade if a critical vulnerability is found.
                    if adversarial_loss.item() < -0.7:
                        self._propose_architectural_upgrade("dynamic_quantization_upgrade")

                    arlc.self_correct_from_failure(failure_report, self.model)

    def propose_new_relationship_from_failure(self, failure_report: Dict[str, Any]):
        relationship_name = f"Vulnerability_{random.randint(100, 999)}"
        description = f"An emergent relationship identified by the Adversarial Module as a vulnerability. It describes a conceptual link that, when perturbed, causes the model's output to diverge significantly."
        self.ckg.propose_new_relationship_type(relationship_name, description, is_vulnerability=True)
        print(f"Adversarial Module proposed new conceptual relationship: '{relationship_name}'")

    # New Method: Proposes an architectural upgrade and logs it to the CKG.
    def _propose_architectural_upgrade(self, upgrade_type: str):
        proposal_id = f"proposal_{self.ckg.proposal_counter}"
        proposal = {
            "proposal_id": proposal_id,
            "proposer": "AdversarialModule",
            "type": upgrade_type,
            "predicted_impact": self.arch_layer.predict_impact(upgrade_type),
            "proposed_changes": self.arch_layer.get_proposed_changes(upgrade_type),
            "status": "pending_human_review",
            "timestamp": datetime.now().isoformat()
        }
        self.ckg.add_node(proposal_id, proposal)
        self.ckg.add_edge("AdversarialModule", proposal_id, "PROPOSED_UPGRADE")
        print(f"\n[Adversary] Proposed a new architectural upgrade: '{upgrade_type}'")
        print(f"Waiting for human review via the Explainability Module.")

