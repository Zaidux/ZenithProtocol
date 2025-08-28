# /src/training/federated_learning.py

import torch
import random
from typing import Dict, List, Tuple
from ..utils.config import Config
from ..models.asreh_model import ASREHModel
from ..conceptual_knowledge_graph.ckg import ConceptualKnowledgeGraph
from ..models.adversarial_module import AdversarialModule # New Import
from ..models.arlc_controller import ARLCController # New Import
from ..models.explainability_module import ExplainabilityModule # New Import
from ..models.sswm import SSWM # New Import

class FederatedLearner:
    """
    Simulates a federated learning server. It orchestrates the training rounds,
    aggregates model updates from clients, and manages the global model.
    It now integrates with the CKG, Meta-Learning, and an Adversarial Module.
    """
    def __init__(self, global_model: ASREHModel, clients: List[ASREHModel], criterion, optimizer, ckg: ConceptualKnowledgeGraph):
        self.global_model = global_model
        self.clients = clients
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = Config()
        self.ckg = ckg # New: Integrated CKG
        
        # New: Adversarial defense for the federated server
        self.adversary = AdversarialModule(global_model) 
        self.em = ExplainabilityModule(global_model, SSWM(input_dim=1), ckg) # Simplified SSWM for EM init
        self.arlc = ARLCController(strategic_planner=None, sswm=None, ckg=ckg) # Simplified ARLC for self-correction

    def select_clients(self, num_clients: int) -> List[ASREHModel]:
        """Randomly selects a subset of clients for the current training round."""
        return random.sample(self.clients, num_clients)

    def train_client_model(self, client_model: ASREHModel, data: List[Dict]) -> Tuple[ASREHModel, List[Dict]]:
        """
        Simulates on-device training for a single client.
        It also collects new conceptual discoveries to be sent back to the server.
        """
        client_model.train()
        client_optimizer = torch.optim.Adam(client_model.parameters(), lr=self.config.LEARNING_RATE)
        
        discovered_concepts = []
        
        for episode in data:
            state, conceptual_features, target = episode['state'], episode['conceptual_features'], episode['target']
            domain = episode['domain']

            state_tensor = torch.tensor(state).unsqueeze(0).float().to(self.config.DEVICE)
            conceptual_tensor = torch.tensor(conceptual_features).unsqueeze(0).float().to(self.config.DEVICE)
            target_tensor = torch.tensor(target).unsqueeze(0).float().to(self.config.DEVICE)

            predicted_output, fused_representation, moe_loss = client_model(state_tensor, conceptual_tensor, domain)

            # Check for new conceptual discoveries made by the client
            _, new_concept_name = client_model.hct_layer.cde.analyze_for_new_concepts(fused_representation, reward=1.0, domain=domain)
            if new_concept_name:
                discovered_concepts.append({'concept': new_concept_name, 'domain': domain})

            loss = self.criterion(predicted_output, target_tensor) + moe_loss

            client_optimizer.zero_grad()
            loss.backward()
            client_optimizer.step()

        print(f"Client model trained on {len(data)} data points.")
        return client_model, discovered_concepts

    def aggregate_updates(self, client_models: List[ASREHModel], all_discovered_concepts: List[Dict]) -> None:
        """
        Aggregates the weights from the trained client models and
        updates the CKG with new concepts.
        """
        # New: Use the Adversarial Module to scrutinize client updates
        sanitized_models = []
        for client_model in client_models:
            is_malicious = self.adversary.check_for_malicious_update(self.global_model.state_dict(), client_model.state_dict())
            if is_malicious:
                print("[Federated Server] Malicious update detected from a client. Rejecting update.")
                failure_report = self.em.analyze_and_report_failure(
                    original_input=torch.randn(1, self.config.HCT_DIM),
                    adversarial_input=torch.randn(1, self.config.HCT_DIM),
                    original_output=torch.randn(1, self.config.HCT_DIM),
                    adversarial_output=torch.randn(1, self.config.HCT_DIM)
                )
                self.arlc.self_correct_from_failure(failure_report, self.global_model)
            else:
                sanitized_models.append(client_model)

        if not sanitized_models:
            print("[Federated Server] All client updates were rejected.")
            return

        global_state_dict = self.global_model.state_dict()
        sum_of_weights = {name: torch.zeros_like(param) for name, param in global_state_dict.items()}

        for client_model in sanitized_models:
            client_state_dict = client_model.state_dict()
            for name, param in client_state_dict.items():
                sum_of_weights[name] += param

        num_selected_clients = len(sanitized_models)
        averaged_weights = {name: param / num_selected_clients for name, param in sum_of_weights.items()}

        self.global_model.load_state_dict(averaged_weights)
        print("Global model updated via federated averaging.")
        
        # New: Update the CKG with newly discovered concepts from clients
        for concept_info in all_discovered_concepts:
            self.ckg.add_node(concept_info['concept'], {"type": "emergent_concept", "source": "federated_learning", "domain": concept_info['domain']})
        print(f"CKG updated with {len(all_discovered_concepts)} new conceptual discoveries.")

    def run_federated_training(self, client_data: List[List[Dict]]):
        """
        Runs the full federated learning training loop.
        """
        print(f"Starting Federated Learning for {self.config.FL_ROUNDS} rounds.")
        
        # Initialize client models with meta-learned weights from the global model
        for client in self.clients:
            client.load_state_dict(self.global_model.state_dict())
        
        for round_num in range(self.config.FL_ROUNDS):
            print(f"\n--- Federated Learning Round {round_num + 1}/{self.config.FL_ROUNDS} ---")

            selected_clients = self.select_clients(self.config.CLIENTS_PER_ROUND)
            
            trained_client_models = []
            all_discovered_concepts = []
            for client_idx, client in enumerate(selected_clients):
                print(f"Training client {client_idx + 1}...")
                client_data_subset = client_data[client_idx]
                trained_model, discovered_concepts = self.train_client_model(client, client_data_subset)
                trained_client_models.append(trained_model)
                all_discovered_concepts.extend(discovered_concepts)

            self.aggregate_updates(trained_client_models, all_discovered_concepts)

        print("\nFederated Learning complete. The global model is now a synthesis of distributed knowledge.")
