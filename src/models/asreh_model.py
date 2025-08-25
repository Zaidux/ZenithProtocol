# /src/model/asreh_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

class SelfAttention(nn.Module):
    """
    Implements a scaled dot-product attention mechanism.
    This allows the model to weigh different parts of the input more heavily.
    """
    def __init__(self, d_model):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_model ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, v)
        return output, attention_weights

class ASREHModel(nn.Module):
    def __init__(self, hct_dim=64):
        super(ASREHModel, self).__init__()
        
        # --- Shared Visual/State Encoder (Split Mind) ---
        self.shared_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # --- Self-Attention Layer ---
        # We need a size-agnostic way to apply attention
        self.attention = SelfAttention(d_model=64)
        
        # --- Conceptual Encoder (The "Understanding is Key" component) ---
        # This branch processes abstract, conceptual features from the environment.
        # It's flexible to handle different numbers of features from different games.
        self.conceptual_encoder = nn.Sequential(
            nn.Linear(5, 32), # 5 features from chess (or dynamically change)
            nn.ReLU(),
            nn.Linear(32, hct_dim)
        )
        
        # --- Hyper-Conceptual Thinking (HCT) Core ---
        # This layer fuses the visual state with the abstract conceptual knowledge.
        self.hct_core = nn.Sequential(
            nn.Linear(hct_dim + 64 * 5 * 2, 256),  # Assuming 5x2 is final feature map size
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # --- Domain-Specific Decoders ---
        # Each game has a specific output head.
        self.tetris_decoder = nn.Linear(128, 10)  # Output for 10 possible columns
        self.chess_decoder = nn.Linear(128, 64)   # Simplified for 64 possible squares

    def forward(self, state, conceptual_features, domain: str):
        # Pass state through the shared encoder
        x = self.shared_encoder(state)
        
        # Reshape for attention and apply
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, -1).transpose(1, 2)
        x, attention_weights = self.attention(x)
        
        # Flatten the output for the HCT core
        x = x.view(batch_size, -1)
        
        # Process conceptual features
        conceptual_embedding = self.conceptual_encoder(conceptual_features)
        
        # Fuse the two branches in the HCT core
        fused_representation = torch.cat((x, conceptual_embedding), dim=1)
        fused_output = self.hct_core(fused_representation)
        
        if domain == 'tetris':
            output = self.tetris_decoder(fused_output)
            return output, attention_weights
        elif domain == 'chess':
            output = self.chess_decoder(fused_output)
            return output, attention_weights
        else:
            raise ValueError("Invalid domain specified. Choose 'tetris' or 'chess'.")
      
