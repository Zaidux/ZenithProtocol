# /src/zaie_chatbot/model/zaie_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Tuple, Any

# Import the core Zenith Protocol components
from models.arlc_controller import ARLCController
from models.explainability_module import ExplainabilityModule
from models.sswm import SSWM
from models.strategic_planner import StrategicPlanner
from conceptual_knowledge_graph.ckg import ConceptualKnowledgeGraph
from web_access.web_access import WebAccess

# Import the C++ backends
import conceptual_encoder_cpp
import moe_router_cpp
import asreh_model_cpp

class ZAIEModel(nn.Module):
    """
    The Ziver Adaptive Intelligence Engine (ZAIE) is the commercial embodiment of the
    Zenith Protocol's core principles for conversational AI.
    It is a high-level orchestrator that composes C++-optimized modules for maximum efficiency.
    """
    def __init__(self,
                 vocab_size: int,
                 input_dim: int,
                 hct_dim: int,
                 num_experts: int,
                 num_heads: int,
                 ckg: ConceptualKnowledgeGraph,
                 web_access: WebAccess):
        
        super().__init__()
        
        # --- Core Zenith Components (Python Orchestrators) ---
        self.ckg = ckg
        self.web_access = web_access
        self.sswm = SSWM(input_dim=hct_dim, hidden_dim=hct_dim, ckg=ckg, web_access=web_access)
        self.strategic_planner = StrategicPlanner()
        
        # We pass a reference to the model to the ARLC and EM for self-correction and explainability
        # This is a circular dependency, but it's a core part of the self-regulating design
        self.arlc = ARLCController(
            strategic_planner=self.strategic_planner,
            sswm=self.sswm,
            ckg=self.ckg,
            web_access=self.web_access,
            model=self
        )
        self.explainability_module = ExplainabilityModule(
            model=self,
            sswm=self.sswm,
            ckg=self.ckg
        )
        
        # --- C++ Optimized Backends ---
        self.cpp_conceptual_encoder = conceptual_encoder_cpp.ConceptualEncoder()
        self.cpp_moe_router = moe_router_cpp.ConceptualAwareRouter(
            input_dim=hct_dim,
            num_experts=num_experts,
            top_k=2
        )
        self.cpp_asreh_model = asreh_model_cpp.ASREHModel(
            in_channels=1, # This is a placeholder as ZAIE is text-based
            hct_dim=hct_dim,
            num_experts=num_experts
        )
        
        # --- Model Layers for Conversational AI ---
        self.token_embedding = nn.Embedding(vocab_size, hct_dim)
        self.output_layer = nn.Linear(hct_dim, vocab_size)

        # The ASREHModel's components are now composed directly here,
        # rather than through inheritance.
        
    def forward(self, text_input: torch.Tensor, conceptual_input: torch.Tensor, is_training: bool = True) -> Tuple[torch.Tensor, Any]:
        """
        The forward pass of the ZAIE model.
        
        Args:
            text_input (Tensor): A tensor of token IDs.
            conceptual_input (Tensor): A tensor of conceptual feature vectors.
            is_training (bool): Flag for training vs. inference.
            
        Returns:
            A tuple containing:
            - output_logits (Tensor): The raw logits for the next token.
            - decision_context (Dict): A conceptual dictionary for the EM.
        """
        # 1. Process text input and conceptual input.
        text_embedded = self.token_embedding(text_input)
        
        # 2. Fuse the two inputs using the C++ backend for the conceptual attention.
        # We're passing the text embeddings as the "visual" features for the ASREH model's fusion.
        fused_representation_np = self.cpp_asreh_model.forward(
            text_embedded.detach().cpu().numpy(),
            conceptual_input.detach().cpu().numpy()
        )
        fused_representation = torch.from_numpy(fused_representation_np)
        
        # 3. Use the C++ MoE router to select the conversational expert.
        # We need a conceptual context to feed to the router.
        conceptual_context = moe_router_cpp.ConceptualContext()
        conceptual_context.context_map = {'topic': ['conversation']}
        
        top_k_indices_np = self.cpp_moe_router.route(
            fused_representation.detach().cpu().numpy(),
            conceptual_context
        )
        top_k_indices = torch.from_numpy(top_k_indices_np).long()

        # For a simplified example, we'll assume a single expert is selected.
        # In a real system, we'd use the top_k_indices to route to different experts.
        moe_output = fused_representation # Placeholder for the expert's output
        
        # 4. Generate the final output logits.
        output_logits = self.output_layer(moe_output)
        
        # 5. Prepare the conceptual context for the Explainability Module.
        decision_context = {
            'chosen_expert': top_k_indices[0].item(),
            'conceptual_factors': ['conversation', 'dialogue']
        }
        
        return output_logits, decision_context

