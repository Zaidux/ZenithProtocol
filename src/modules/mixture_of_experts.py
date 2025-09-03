# /src/modules/mixture_of_experts.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from ..conceptual_knowledge_graph.ckg import ConceptualKnowledgeGraph # New import

class MixtureOfExperts(nn.Module):
    """
    A Mixture of Experts (MoE) layer with a conceptual-aware router.
    The 'router' network selects and weighs the outputs of several 'expert' networks
    based on the input and additional conceptual context from the CKG.
    """
    def __init__(self, input_dim: int, num_experts: int, top_k: int = 2, ckg: ConceptualKnowledgeGraph = None):
        super(MixtureOfExperts, self).__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.ckg = ckg # The Conceptual Knowledge Graph instance
        
        assert self.top_k <= self.num_experts, "top_k cannot be greater than num_experts"

        # The router now takes the input and a conceptual context vector from the CKG
        # This allows it to make a more informed, conceptually-aware routing decision.
        self.router = nn.Linear(input_dim, num_experts)
        
        # Experts - a list of specialized neural networks
        self.experts = nn.ModuleList([
            nn.Linear(input_dim, input_dim) for _ in range(num_experts)
        ])
        
        # A learnable parameter to weigh the conceptual context's influence on routing.
        self.context_weight = nn.Parameter(torch.rand(1))

    def forward(self, x: torch.Tensor, conceptual_context: Dict[str, List[str]]):
        """
        Forward pass with conceptual-aware routing.
        Args:
            x: Input tensor.
            conceptual_context: A dictionary of concepts and their properties from the CKG.
                                e.g., {'topic': ['chess', 'game'], 'reason': ['strategy']}
        """
        # 1. Conceptual-Aware Router: Combine input with CKG context
        # We simulate getting a conceptual vector from the CKG.
        # In a real system, the CKG would return a vector representation of the context.
        context_vector = self._get_context_vector(conceptual_context)
        
        # The router now learns to weigh the input and the conceptual context.
        # This gives it a more grounded understanding of the "why" behind the input.
        gate_logits = self.router(x) + self.context_weight * context_vector
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
            current_expert_output = self.experts[expert_idx](x)
            expert_outputs += expert_weight * current_expert_output
        
        # 4. Load Balancing Loss
        # This is a critical component to prevent expert underutilization.
        load_balancing_loss = self._calculate_conceptual_load_balancing_loss(weights, top_k_weights, top_k_indices)

        return expert_outputs, load_balancing_loss

    def _get_context_vector(self, conceptual_context: Dict[str, List[str]]) -> torch.Tensor:
        """
        Simulates querying the CKG and converting the results into a vector.
        In a real-world scenario, this would involve a sophisticated retrieval mechanism.
        """
        vector = torch.zeros(self.num_experts)
        # We'll map the concepts from the context to specific experts.
        # This is a placeholder for a more complex routing policy.
        concept_expert_map = {
            "chess": 0, "game": 1, "tetris": 2, "finance": 3, "law": 4, "medicine": 5, "strategy": 0, "knowledge": 1
        }
        
        for key, concepts in conceptual_context.items():
            for concept in concepts:
                if concept in concept_expert_map:
                    expert_id = concept_expert_map[concept]
                    vector[expert_id] += 1.0 # Boost the score for the relevant expert
        
        return vector

    def _calculate_conceptual_load_balancing_loss(self, weights: torch.Tensor, top_k_weights: torch.Tensor, top_k_indices: torch.Tensor) -> torch.Tensor:
        """
        Calculates a load-balancing loss to encourage balanced expert usage.
        Based on the idea of penalizing a router that routes excessive tokens to a few experts.
        """
        # The average router probability for each expert
        avg_router_prob = weights.mean(dim=0)
        
        # The fraction of tokens routed to each expert
        # Create a mask for the top-k experts that were selected
        expert_assignment_mask = F.one_hot(top_k_indices, self.num_experts).sum(dim=1).float()
        fraction_of_tokens_routed = expert_assignment_mask.mean(dim=0)

        # A loss term that penalizes the router if the average probability
        # to an expert is much higher than the actual number of tokens routed.
        # This encourages the router to actually use the experts it thinks are best.
        load_loss = (avg_router_prob * fraction_of_tokens_routed).sum()

        return load_loss

if __name__ == '__main__':
    # Initialize a mock CKG for the example.
    from ..conceptual_knowledge_graph.ckg import ConceptualKnowledgeGraph
    ckg = ConceptualKnowledgeGraph()
    
    # Example usage of the MoE module
    moe_layer = MixtureOfExperts(input_dim=128, num_experts=8, top_k=2, ckg=ckg)
    
    # Create a mock input tensor and conceptual context from the CKG
    input_tensor = torch.randn(1, 128)
    # The conceptual context would come from a query to the CKG after the input is processed.
    conceptual_context_example = {'topic': ['chess', 'strategy']}
    
    output, loss = moe_layer(input_tensor, conceptual_context_example)
    print(f"Output shape: {output.shape}")
    print(f"Load balancing loss: {loss.item()}")
