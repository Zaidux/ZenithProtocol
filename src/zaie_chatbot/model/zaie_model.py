# /src/zaie_chatbot/model/zaie_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

# Assuming ASREHModel and other core components are in src/models
from models.asreh_model import ASREHModel
from models.arlc_controller import ARLC
from models.explainability_module import ExplainabilityModule
from models.spl_model import SplitMind
from models.sswm_model import SSWM

class ZAIEModel(ASREHModel):
    def __init__(self, vocab_size, conceptual_embedding_size,
                 hidden_dim, num_experts, num_heads, num_layers):
        
        # We call the parent constructor to inherit all the core components.
        # The ASREHModel will initialize the ConceptualAttentionLayer,
        # SplitMind (MoE), ARLC, and EM.
        super().__init__(
            conceptual_embedding_size=conceptual_embedding_size,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_heads=num_heads,
            num_layers=num_layers
        )
        
        # --- New Components for Conversational AI ---
        # 1. Token Embeddings: Converts text tokens into dense vectors.
        # This is the "sensory" input.
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)

        # 2. Conceptual Encoder: Processes the structured conceptual data.
        # This is the "conceptual" input.
        self.conceptual_encoder = nn.Linear(conceptual_embedding_size, hidden_dim)

        # 3. Output Layer: Maps the final hidden state to a vocabulary
        # for generating text.
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

        # 4. Split Mind Module for text (MoE)
        self.split_mind = SplitMind(input_dim=hidden_dim, num_experts=num_experts)

        # 5. Strategic Planner (Placeholder for now)
        self.strategic_planner = nn.Linear(hidden_dim, num_experts) # Router

    def forward(self, text_input, conceptual_input, is_training=True):
        """
        The forward pass of the ZAIE model.
        
        Args:
            text_input (Tensor): A tensor of token IDs representing the user's message.
            conceptual_input (Tensor): A tensor of conceptual feature vectors.
            is_training (bool): Flag to control behavior during training vs. inference.
        
        Returns:
            A tuple containing:
            - output_logits (Tensor): The raw logits for the next token.
            - conceptual_reasoning (Tensor): A conceptual vector used by the EM.
        """
        # 1. Process text input and conceptual input in parallel.
        # The text is our new "visual" data.
        text_embedded = self.token_embedding(text_input)
        
        # The conceptual input is our "conceptual" data.
        conceptual_encoded = self.conceptual_encoder(conceptual_input)

        # 2. The Conceptual Attention Layer fuses the two.
        # This is the core of our "Understanding is Key" principle.
        fused_output, attn_weights = self.conceptual_attention_layer(text_embedded, conceptual_encoded)
        
        # 3. Use the Split Mind (MoE) to handle different conversational domains.
        # The strategic_planner acts as a router.
        expert_weights = F.softmax(self.strategic_planner(fused_output), dim=-1)
        moe_output = self.split_mind(fused_output, expert_weights)

        # 4. Use the ARLC to guide the model towards a better response.
        # In this context, the ARLC's "reward" is the conversational score.
        # For simplicity in this forward pass, we'll assume ARLC provides
        # a final hidden state.
        final_hidden_state = moe_output

        # 5. Generate the final output logits.
        output_logits = self.output_layer(final_hidden_state)
        
        # The EM will analyze the final hidden state and attention weights
        # to generate an explanation.
        conceptual_reasoning = attn_weights

        return output_logits, conceptual_reasoning


class ARLC(nn.Module):
    # This is a placeholder ARLC to make the code runnable.
    def forward(self, input):
        return input

class ExplainabilityModule(nn.Module):
    # This is a placeholder EM to make the code runnable.
    def forward(self, conceptual_reasoning):
        return "Explanation based on conceptual reasoning."

class SplitMind(nn.Module):
    # This is a placeholder SplitMind to make the code runnable.
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_experts)])

    def forward(self, input, weights):
        # A simple weighted sum of expert outputs
        expert_outputs = torch.stack([expert(input) for expert in self.experts], dim=0)
        return torch.sum(expert_outputs * weights.unsqueeze(0), dim=0)

class SSWM(nn.Module):
    # This is a placeholder SSWM to make the code runnable.
    def forward(self, input):
        return input

class ASREHModel(nn.Module):
    # This is a minimal placeholder to allow ZAIEModel to inherit.
    def __init__(self, conceptual_embedding_size, hidden_dim, num_experts, num_heads, num_layers):
        super().__init__()
        self.conceptual_attention_layer = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )


