# /src/training/federated_learning.py

import torch
import random
from typing import Dict, List, Tuple
from ..utils.config import Config
from ..models.asreh_model import ASREHModel

class FederatedLearner:
    """
    Simulates a federated learning server. It orchestrates the training rounds,
    aggregates model updates from clients, and manages the global model.
    """
    def __init__(self, global_model: ASREHModel, clients: List[ASREHModel], criterion, optimizer):
        self.global_model = global_model
        self.clients = clients
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = Config()

    def select_clients(self, num_clients: int) -> List[ASREHModel]:
        """Randomly selects a subset of clients for the current training round."""
        return random.sample(self.clients, num_clients)

    def train_client_model(self, client_model: ASREHModel, data: List[Dict]) -> ASREHModel:
        """
        Simulates on-device training for a single client.
        The client trains its local model on its own data.
        """
        client_model.train()
        client_optimizer = torch.optim.Adam(client_model.parameters(), lr=self.config.LEARNING_RATE)
        
        # Simulate a small local training epoch
        for episode in data:
            state, conceptual_features, target = episode['state'], episode['conceptual_features'], episode['target']
            domain = episode['domain']
            
            # Convert to tensors and move to device
            state_tensor = torch.tensor(state).unsqueeze(0).float().to(self.config.DEVICE)
            conceptual_tensor = torch.tensor(conceptual_features).unsqueeze(0).float().to(self.config.DEVICE)
            target_tensor = torch.tensor(target).unsqueeze(0).float().to(self.config.DEVICE)
            
            # Forward pass
            predicted_output, _, _ = client_model(state_tensor, conceptual_tensor, domain)
            
            # Calculate loss (example)
            loss = self.criterion(predicted_output, target_tensor)
            
            # Backpropagation
            client_optimizer.zero_grad()
            loss.backward()
            client_optimizer.step()
            
        print(f"Client model trained on {len(data)} data points.")
        return client_model

    def aggregate_updates(self, client_models: List[ASREHModel]) -> None:
        """
        Aggregates the weights from the trained client models using federated averaging.
        """
        global_state_dict = self.global_model.state_dict()
        
        # Initialize an empty dictionary for the sum of all client weights
        sum_of_weights = {name: torch.zeros_like(param) for name, param in global_state_dict.items()}
        
        # Aggregate weights
        for client_model in client_models:
            client_state_dict = client_model.state_dict()
            for name, param in client_state_dict.items():
                sum_of_weights[name] += param
        
        # Average the weights
        num_selected_clients = len(client_models)
        averaged_weights = {name: param / num_selected_clients for name, param in sum_of_weights.items()}
        
        # Update the global model with the new averaged weights
        self.global_model.load_state_dict(averaged_weights)
        print("Global model updated via federated averaging.")

    def run_federated_training(self, client_data: List[List[Dict]]):
        """
        Runs the full federated learning training loop.
        """
        print(f"Starting Federated Learning for {self.config.FL_ROUNDS} rounds.")
        
        for round_num in range(self.config.FL_ROUNDS):
            print(f"\n--- Federated Learning Round {round_num + 1}/{self.config.FL_ROUNDS} ---")
            
            # Select a subset of clients for this round
            selected_clients = self.select_clients(self.config.CLIENTS_PER_ROUND)
            
            trained_client_models = []
            for client_idx, client in enumerate(selected_clients):
                print(f"Training client {client_idx + 1}...")
                client_data_subset = client_data[client_idx] # Assume data is pre-partitioned
                trained_model = self.train_client_model(client, client_data_subset)
                trained_client_models.append(trained_model)
            
            # Aggregate the updates
            self.aggregate_updates(trained_client_models)

        print("\nFederated Learning complete.")
