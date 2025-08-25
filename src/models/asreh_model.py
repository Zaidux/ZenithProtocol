# /src/models/asreh_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

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
                 num_tetris_features: int = 4,
                 num_chess_features: int = 5,
                 hct_dim: int = 64):

        super(ASREHModel, self).__init__()
        self.hct_dim = hct_dim
        self.in_channels = in_channels

        # --- Shared Visual/State Encoder (The "Eye") ---
        # This encoder learns a generalized representation from both games
        self.shared_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, hct_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # --- Conceptual Encoders (The "Understanding" component) ---
        # Each domain has its own conceptual encoder
        self.tetris_conceptual_encoder = nn.Sequential(
            nn.Linear(num_tetris_features, hct_dim),
            nn.ReLU(),
            nn.Linear(hct_dim, hct_dim)
        )
        self.chess_conceptual_encoder = nn.Sequential(
            nn.Linear(num_chess_features, hct_dim),
            nn.ReLU(),
            nn.Linear(hct_dim, hct_dim)
        )

        # --- Conceptual Attention Layer (The Core of Zenith) ---
        self.conceptual_attention = ConceptualAttention(embed_dim=hct_dim, num_heads=4)

        # --- Domain-Specific Decoders ---
        self.tetris_decoder = nn.Sequential(
            nn.Linear(hct_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 10 * 20),
            nn.Sigmoid()
        )
        self.chess_decoder = nn.Sequential(
            nn.Linear(hct_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * 64) # A simplified output for move prediction
        )

    def forward(self, state: torch.Tensor, conceptual_features: torch.Tensor, domain: str):
        # Pass state through the shared encoder
        x = self.shared_encoder(state)

        # Flatten the encoder output
        batch_size = x.size(0)

        # Use a flexible size for the flattened tensor
        visual_features = x.view(batch_size, self.hct_dim, -1).transpose(1, 2)

        # Process conceptual features with the correct domain-specific encoder
        if domain == 'tetris':
            conceptual_embedding = self.tetris_conceptual_encoder(conceptual_features)
        elif domain == 'chess':
            conceptual_embedding = self.chess_conceptual_encoder(conceptual_features)
        else:
            raise ValueError("Invalid domain specified.")

        # Fuse the two branches using Conceptual Attention
        fused_representation = self.conceptual_attention(visual_features, conceptual_embedding)

        # Choose the correct decoder
        if domain == 'tetris':
            predicted_board = self.tetris_decoder(fused_representation)
            return predicted_board.view(batch_size, 1, 20, 10), fused_representation
        elif domain == 'chess':
            predicted_move_logits = self.chess_decoder(fused_representation)
            return predicted_move_logits, fused_representation
