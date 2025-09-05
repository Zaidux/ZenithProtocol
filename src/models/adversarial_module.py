# /src/models/adversarial_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Dict, List, Tuple
from ..utils.config import Config
from ..conceptual_knowledge_graph.ckg import ConceptualKnowledgeGraph

class AdversarialModule(nn.Module):
    """
    The Adversarial Module is a dedicated model that generates inputs designed
    to confuse the ASREHModel. It learns to find and exploit weaknesses in the
    model's reasoning, forcing the system to self-correct.
    """
    def __init__(self, model: nn.Module, ckg: ConceptualKnowledgeGraph):
        super(AdversarialModule, self).__init__()
        self.model = model
        self.ckg = ckg # New: CKG instance for proposing new relationships
        self.config = Config()

        # A simple generator network to create adversarial noise or a new state
        self.generator = nn.Sequential(
            nn.Linear(self.config.HCT_DIM, self.config.HCT_DIM * 2),
            nn.ReLU(),
            nn.Linear(self.config.HCT_DIM * 2, self.config.HCT_DIM),
        ).to(self.config.DEVICE)

        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.config.ADVERSARIAL_LR)

        # A loss function that rewards finding errors
        self.adversarial_loss_fn = nn.MSELoss()

    def generate_adversarial_input(self, fused_representation: torch.Tensor) -> torch.Tensor:
        """
        Generates an adversarial perturbation or a new state to confuse the model.
        The perturbation is designed to make the model's prediction diverge.
        """
        # The generator learns to produce an offset that maximizes the model's loss
        perturbation = self.generator(fused_representation)
        adversarial_input = fused_representation + perturbation
        return adversarial_input

    def run_adversarial_training(self, arlc, em):
        """
        Runs the adversarial training loop. The adversary and the main model
        are trained in a zero-sum game.
        """
        print("\n[Adversary] Starting adversarial training loop...")
        self.generator.train()
        self.model.eval() # The main model is a fixed target during this phase

        for epoch in range(self.config.ADVERSARIAL_EPOCHS):
            # 1. Get a sample input from the model's normal operation
            random_fused_rep = torch.randn(1, self.config.HCT_DIM).to(self.config.DEVICE)

            # 2. Generate an adversarial version of that input
            adversarial_rep = self.generate_adversarial_input(random_fused_rep)

            # 3. Pass both inputs to the main model and compare predictions
            original_output, _, _ = self.model.forward(random_fused_rep, torch.randn(1, 64), 'tetris')
            adversarial_output, _, _ = self.model.forward(adversarial_rep, torch.randn(1, 64), 'tetris')

            # 4. Calculate the adversarial loss
            adversarial_loss = -self.adversarial_loss_fn(original_output, adversarial_output)

            # 5. Backpropagate to train the generator
            self.optimizer.zero_grad()
            adversarial_loss.backward()
            self.optimizer.step()

            if (epoch + 1) % self.config.ADVERSARIAL_LOG_INTERVAL == 0:
                print(f"[Adversary] Epoch {epoch+1}/{self.config.ADVERSARIAL_EPOCHS}, Adversarial Loss: {adversarial_loss.item():.4f}")

                if adversarial_loss.item() < -0.5:
                    print("\n[Adversary] Major weakness found! Generating failure report...")
                    # 6. Use the Explainability Module to analyze the failure
                    failure_report = em.analyze_and_report_failure(
                        original_input=random_fused_rep,
                        adversarial_input=adversarial_rep,
                        original_output=original_output,
                        adversarial_output=adversarial_output
                    )
                    
                    # New: Propose a new relationship type based on the failure
                    self.propose_new_relationship_from_failure(failure_report)
                    
                    # 7. ARLC takes the report and performs self-correction
                    arlc.self_correct_from_failure(failure_report, self.model)

    def propose_new_relationship_from_failure(self, failure_report: Dict[str, Any]):
        """
        Analyzes a failure report and proposes a new conceptual relationship to the CKG.
        """
        # This is a placeholder for a more sophisticated analysis.
        # In a real system, the module would analyze the conceptual differences
        # between the original and adversarial inputs to identify a missing link.
        
        # New: Propose a new relationship type based on the failure
        relationship_name = f"Vulnerability_{random.randint(100, 999)}"
        description = f"An emergent relationship identified by the Adversarial Module as a vulnerability. It describes a conceptual link that, when perturbed, causes the model's output to diverge significantly."
        self.ckg.propose_new_relationship_type(relationship_name, description, is_vulnerability=True)
        print(f"Adversarial Module proposed new conceptual relationship: '{relationship_name}'")
