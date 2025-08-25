# /src/models/sswm.py

import torch
import torch.nn as nn
from typing import Dict, Tuple

class SelfSupervisedWorldModel(nn.Module):
    """
    The Self-Supervised World Model (SSWM) is a predictive network that learns to
    anticipate the consequences of actions. It predicts the next state and reward.
    """
    def __init__(self, fused_rep_dim: int = 64):
        super(SelfSupervisedWorldModel, self).__init__()
        self.fused_rep_dim = fused_rep_dim

        # Predicts the next fused representation
        self.state_predictor = nn.Sequential(
            nn.Linear(fused_rep_dim, 128),
            nn.ReLU(),
            nn.Linear(128, fused_rep_dim)
        )

        # Predicts the next reward
        self.reward_predictor = nn.Sequential(
            nn.Linear(fused_rep_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, fused_representation: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Takes the current fused representation and a one-hot encoded action to
        predict the next representation and a reward.
        """
        # For simplicity, we'll combine the fused representation with the action
        # In a more complex model, this would be more sophisticated.
        combined_input = fused_representation + action

        # Predict the next state and reward
        predicted_next_rep = self.state_predictor(combined_input)
        predicted_reward = self.reward_predictor(combined_input)

        return predicted_next_rep, predicted_reward

    def simulate_what_if(self,
                         current_fused_rep: torch.Tensor,
                         action: int,
                         domain: str,
                         num_steps: int = 1) -> Dict:
        """
        Simulates a hypothetical scenario and predicts the outcome.
        This is the counterfactual component.
        """
        # Create a one-hot encoded tensor for the action
        action_tensor = torch.zeros_like(current_fused_rep)
        action_tensor[0, action] = 1.0 # Assuming a single action per step

        with torch.no_grad():
            simulated_rep = current_fused_rep
            total_predicted_reward = 0.0
            
            for _ in range(num_steps):
                predicted_next_rep, predicted_reward = self.forward(simulated_rep, action_tensor)
                simulated_rep = predicted_next_rep.clone()
                total_predicted_reward += predicted_reward.item()

        return {
            'action': action,
            'predicted_reward': total_predicted_reward,
            'predicted_next_state_rep': simulated_rep
        }
