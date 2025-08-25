# /src/models/asreh_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from ..modules.mixture_of_experts import MixtureOfExperts

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
                 num_experts: int = 4, # New parameter for MoE
                 hct_dim: int = 64):

        super(ASREHModel, self).__init__()
        self.hct_dim = hct_dim
        self.in_channels = in_channels
        self.num_experts = num_experts

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
        # A single encoder to handle a variable number of conceptual features
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
        # The decoder is now shared, allowing for domain transfer
        self.decoder = nn.Sequential(
            nn.Linear(hct_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * 64) # A flexible output size
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

        # Pass the fused representation through the Mixture of Experts
        moe_output, gate_loss = self.moe_layer(fused_representation)

        # Pass the MoE output to the shared decoder
        output = self.decoder(moe_output)

        # Adjust the output shape based on the domain
        if domain == 'tetris':
            # This is a placeholder for a more robust solution
            return output.view(batch_size, 1, 20, 10), fused_representation, gate_loss
        elif domain == 'chess':
            return output, fused_representation, gate_loss
        else:
            raise ValueError("Invalid domain specified.")