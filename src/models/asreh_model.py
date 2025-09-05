# /src/models/asreh_model.py

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from ..modules.mixture_of_experts import MixtureOfExperts
from copy import deepcopy

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

        self.ckg = ckg
        self.web_access = web_access

        self.cpp_asreh_model = asreh_model_cpp.ASREHModel(
            in_channels=in_channels,
            hct_dim=hct_dim,
            num_experts=num_experts
        )
        self.cpp_moe_router = moe_router_cpp.ConceptualAwareRouter(
            input_dim=hct_dim,
            num_experts=num_experts,
            top_k=2
        )

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

        self.decoder = nn.Sequential(
            nn.Linear(hct_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * 64)
        )

    def forward(self, state: torch.Tensor, conceptual_features: torch.Tensor, domain: str):
        fused_representation_np = self.cpp_asreh_model.forward(
            state.detach().cpu().numpy(),
            conceptual_features.detach().cpu().numpy()
        )
        fused_representation = torch.from_numpy(fused_representation_np)

        conceptual_context = moe_router_cpp.ConceptualContext()
        conceptual_context.context_map = {'topic': [domain]}

        top_k_indices_np = self.cpp_moe_router.route(
            fused_representation.detach().cpu().numpy(),
            conceptual_context
        )
        top_k_indices = torch.from_numpy(top_k_indices_np).long()

        moe_output = fused_representation
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
