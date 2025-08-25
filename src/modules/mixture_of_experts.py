# /src/modules/mixture_of_experts.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MixtureOfExperts(nn.Module):
    """
    A Mixture of Experts (MoE) layer.
    The 'router' network learns to select and weigh the outputs of
    several 'expert' networks based on the input.
    """
    def __init__(self, input_dim: int, num_experts: int, top_k: int = 2):
        super(MixtureOfExperts, self).__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k
        assert self.top_k <= self.num_experts

        # Router (Gating Network)
        self.router = nn.Linear(input_dim, num_experts)

        # Experts
        self.experts = nn.ModuleList([
            nn.Linear(input_dim, input_dim) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor):
        # 1. Router computes a probability distribution over experts
        gate_logits = self.router(x)
        weights = F.softmax(gate_logits, dim=-1)

        # 2. Select top_k experts
        top_k_weights, top_k_indices = torch.topk(weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / torch.sum(top_k_weights, dim=-1, keepdim=True)

        # 3. Process input through the selected experts
        expert_outputs = torch.zeros_like(x)
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]
            expert_weight = top_k_weights[:, i].unsqueeze(1)
            
            # This is a simplified expert selection
            current_expert_output = self.experts[i](x)
            expert_outputs += expert_weight * current_expert_output

        # 4. Calculate a load balancing loss to encourage balanced expert usage
        router_prob = F.softmax(gate_logits, dim=-1)
        expert_assignment_mask = F.one_hot(top_k_indices, self.num_experts).sum(dim=1).float()
        load = router_prob.sum(dim=0)
        expert_load = expert_assignment_mask.sum(dim=0)
        
        moe_loss = (router_prob * expert_assignment_mask).sum(dim=1).mean()
        
        return expert_outputs, moe_loss
      
