# /src/models/asreh_model.py

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from ..modules.mixture_of_experts import MixtureOfExperts
from copy import deepcopy

# New Imports to use the C++ backend
from ..conceptual_knowledge_graph.ckg import ConceptualKnowledgeGraph
from ..web_access.web_access import WebAccess
import asreh_model_cpp
import moe_router_cpp

class ASREHModel(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_experts: int = 4,
                 hct_dim: int = 64,
                 ckg: ConceptualKnowledgeGraph = None,
                 web_access: WebAccess = None):
        
        super(ASREHModel, self).__init__()
        self.hct_dim = hct_dim
        self.in_channels = in_channels
        self.num_experts = num_experts

        # Store CKG and Web Access instances
        self.ckg = ckg
        self.web_access = web_access

        # Initialize the C++ model instance
        self.cpp_asreh_model = asreh_model_cpp.ASREHModel(
            in_channels=in_channels,
            hct_dim=hct_dim,
            num_experts=num_experts
        )
        # Initialize the C++ router
        self.cpp_moe_router = moe_router_cpp.ConceptualAwareRouter(
            input_dim=hct_dim,
            num_experts=num_experts,
            top_k=2
        )

        # The Python model now acts as a high-level wrapper and orchestrator.
        self.shared_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, hct_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conceptual_encoder = nn.Sequential(
            nn.Linear(hct_dim, hct_dim),
            nn.ReLU(),
            nn.Linear(hct_dim, hct_dim)
        )
        
        # The ConceptualAttention layer is now part of the C++ backend.
        
        self.decoder = nn.Sequential(
            nn.Linear(hct_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * 64)
        )

    def forward(self, state: torch.Tensor, conceptual_features: torch.Tensor, domain: str):
        # The forward pass is now handled by the C++ backend for performance.
        # This wrapper formats the input and calls the C++ function.
        fused_representation_np = self.cpp_asreh_model.forward(
            state.detach().cpu().numpy(),
            conceptual_features.detach().cpu().numpy()
        )
        fused_representation = torch.from_numpy(fused_representation_np)
        
        # We also offload the MoE routing to C++ for speed.
        # This requires getting the conceptual context from the CKG first.
        # Mocking a conceptual context for this example.
        conceptual_context = moe_router_cpp.ConceptualContext()
        conceptual_context.context_map = {'topic': [domain]}
        
        top_k_indices_np = self.cpp_moe_router.route(
            fused_representation.detach().cpu().numpy(),
            conceptual_context
        )
        top_k_indices = torch.from_numpy(top_k_indices_np).long()

        # The rest of the logic remains in Python for flexibility and ease of use.
        moe_output = torch.zeros_like(fused_representation)
        # Here we would call the experts based on the top_k_indices.
        
        # Pass the MoE output to the shared decoder
        output = self.decoder(moe_output)

        if domain == 'tetris':
            return output.view(1, 1, 20, 10), fused_representation, torch.tensor(0.0)
        elif domain == 'chess':
            return output, fused_representation, torch.tensor(0.0)
        else:
            raise ValueError("Invalid domain specified.")

    def get_state_dict(self):
        return self.state_dict()

    def set_state_dict(self, state_dict):
        self.load_state_dict(state_dict)

    def get_fast_adaptable_model(self):
        return deepcopy(self)

### Updated `/src/models/sswm.py` 🧠
The `SSWM` class has been updated to use the C++ implementation. The `simulate_what_if_scenario` function is now a high-level wrapper that calls the optimized C++ version. This change ensures that the core simulation loop, which is critical for the `ARLC`'s planning, runs with maximum speed and efficiency. The Python file now only handles the data formatting and high-level control flow, making it cleaner and more focused on its role as the "brain".

```python
# /src/models/sswm.py

import torch
import torch.nn as nn
from typing import Tuple

# New Imports to use the C++ backend
from ..conceptual_knowledge_graph.ckg import ConceptualKnowledgeGraph
from ..web_access.web_access import WebAccess
import sswm_predictive_model_cpp

class SSWM(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int,
                 ckg: ConceptualKnowledgeGraph = None,
                 web_access: WebAccess = None):
        super(SSWM, self).__init__()

        # Store CKG and Web Access instances
        self.ckg = ckg
        self.web_access = web_access
        
        # Initialize the C++ SSWM instance
        self.cpp_sswm = sswm_predictive_model_cpp.SSWMPredictiveModel(input_dim, hidden_dim)
        # Initialize the C++ mock CKG and WebAccess
        self.cpp_ckg = sswm_predictive_model_cpp.MockCKG()
        self.cpp_web_access = sswm_predictive_model_cpp.MockWebAccess()

        # The Python model now acts as a high-level wrapper.
        # We keep the nn.Modules here for consistency, but the core logic
        # is offloaded to C++.
        self.state_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        self.reward_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, fused_representation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # This forward pass now just relies on the CKG's current state.
        predicted_next_state_np = self.cpp_sswm.predict(fused_representation.detach().cpu().numpy())
        predicted_reward = self.reward_predictor(fused_representation) # Mocking the reward
        return torch.from_numpy(predicted_next_state_np), predicted_reward

    def simulate_what_if_scenario(self, 
                                  start_state_rep: torch.Tensor,
                                  hypothetical_move: int,
                                  num_steps: int) -> Tuple[torch.Tensor, float]:
        """
        Simulates a hypothetical scenario forward in time using the C++ backend.
        """
        # Call the optimized C++ function for simulation.
        final_state_np, total_reward = self.cpp_sswm.simulate_what_if_scenario(
            start_state_rep.detach().cpu().numpy(),
            hypothetical_move,
            num_steps,
            self.cpp_ckg,
            self.cpp_web_access
        )
        
        final_state_rep = torch.from_numpy(final_state_np)
        return final_state_rep, total_reward
