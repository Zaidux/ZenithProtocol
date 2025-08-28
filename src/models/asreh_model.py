# /src/models/asreh_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from ..modules.mixture_of_experts import MixtureOfExperts
from copy import deepcopy
# New Imports
from ..conceptual_knowledge_graph.ckg import ConceptualKnowledgeGraph
from ..web_access.web_access import WebAccess

class ConceptualAttention(nn.Module):
    """
    Implements a multi-head attention mechanism that fuses visual
    and conceptual embeddings. This layer is the core of the Zenith Protocol.
    """
    def __init__(self, embed_dim: int, num_heads: int):
        super(ConceptualAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, visual_features: torch.Tensor, conceptual_features: torch.Tensor) -> torch.Tensor:
        # Conceptual features act as queries for the visual keys and values
        attn_output, _ = self.multihead_attn(
            query=conceptual_features.unsqueeze(1),
            key=visual_features,
            value=visual_features
        )

        # Add a residual connection and normalize
        fused_output = self.norm(conceptual_features.unsqueeze(1) + attn_output)
        return fused_output.squeeze(1)


class ASREHModel(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_experts: int = 4,
                 hct_dim: int = 64,
                 ckg: ConceptualKnowledgeGraph = None, # New dependency
                 web_access: WebAccess = None):     # New dependency

        super(ASREHModel, self).__init__()
        self.hct_dim = hct_dim
        self.in_channels = in_channels
        self.num_experts = num_experts
        
        # New: Store CKG and Web Access instances
        self.ckg = ckg
        self.web_access = web_access

        # --- Shared Visual/State Encoder (The "Eye") ---
        self.shared_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, hct_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # --- Conceptual Encoder (The "Understanding" component) ---
        # It's now more flexible to handle concepts from the CKG.
        self.conceptual_encoder = nn.Sequential(
            nn.Linear(hct_dim, hct_dim),
            nn.ReLU(),
            nn.Linear(hct_dim, hct_dim)
        )

        # --- Conceptual Attention Layer ---
        self.conceptual_attention = ConceptualAttention(embed_dim=hct_dim, num_heads=4)

        # --- Mixture of Experts Layer ---
        self.moe_layer = MixtureOfExperts(input_dim=hct_dim, num_experts=num_experts)

        # --- Shared Decoder (The "Hand") ---
        self.decoder = nn.Sequential(
            nn.Linear(hct_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * 64)
        )

    def forward(self, state: torch.Tensor, conceptual_features: torch.Tensor, domain: str):
        # Pass state through the shared encoder
        x = self.shared_encoder(state)
        batch_size = x.size(0)
        visual_features = x.view(batch_size, self.hct_dim, -1).transpose(1, 2)

        # Get conceptual features dynamically and pass to encoder
        conceptual_embedding = self.conceptual_encoder(conceptual_features)

        # Fuse the two branches using Conceptual Attention
        fused_representation = self.conceptual_attention(visual_features, conceptual_embedding)

        # New: ARLC will now use the fused representation to guide the search for new knowledge
        # The logic for this is primarily in the ARLC, but the model provides the representation.

        # Pass the fused representation through the Mixture of Experts
        moe_output, gate_loss = self.moe_layer(fused_representation)

        # Pass the MoE output to the shared decoder
        output = self.decoder(moe_output)

        # Adjust the output shape based on the domain
        if domain == 'tetris':
            return output.view(batch_size, 1, 20, 10), fused_representation, gate_loss
        elif domain == 'chess':
            return output, fused_representation, gate_loss
        else:
            raise ValueError("Invalid domain specified.")

    def get_state_dict(self):
        """Returns the model's state dictionary for federated learning."""
        return self.state_dict()

    def set_state_dict(self, state_dict):
        """Loads a state dictionary, useful for receiving a global model."""
        self.load_state_dict(state_dict)

    def get_fast_adaptable_model(self):
        """
        Creates a copy of the model with the current parameters.
        This is used for the inner loop of meta-learning.
        """
        return deepcopy(self)
