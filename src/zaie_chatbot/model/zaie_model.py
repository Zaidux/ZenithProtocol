# /src/zaie_chatbot/model/zaie_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Tuple, Any

from models.arlc_controller import ARLCController
from models.explainability_module import ExplainabilityModule
from models.sswm import SSWM
from models.strategic_planner import StrategicPlanner
from conceptual_knowledge_graph.ckg import ConceptualKnowledgeGraph
from web_access.web_access import WebAccess
from conceptual_encoder.conceptual_encoder import ZenithConceptualEncoder
from conceptual_encoder.conceptual_visual_encoder import ZenithConceptualVisualEncoder
from conceptual_encoder.conceptual_audio_encoder import ZenithConceptualAudioEncoder

import conceptual_encoder_cpp
import moe_router_cpp
import asreh_model_cpp

class ZAIEModel(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 input_dim: int,
                 hct_dim: int,
                 num_experts: int,
                 num_heads: int,
                 ckg: ConceptualKnowledgeGraph,
                 web_access: WebAccess):

        super().__init__()

        self.ckg = ckg
        self.web_access = web_access
        self.sswm = SSWM(input_dim=hct_dim, hidden_dim=hct_dim, ckg=ckg, web_access=web_access)
        self.strategic_planner = StrategicPlanner(model=self, ckg=ckg)

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

        self.text_encoder = ZenithConceptualEncoder(ckg=self.ckg)
        self.visual_encoder = ZenithConceptualVisualEncoder(ckg=self.ckg)
        self.audio_encoder = ZenithConceptualAudioEncoder(ckg=self.ckg)

        self.cpp_conceptual_encoder = conceptual_encoder_cpp.ConceptualEncoder()
        self.cpp_moe_router = moe_router_cpp.ConceptualAwareRouter(
            input_dim=hct_dim,
            num_experts=num_experts,
            top_k=2
        )
        self.cpp_asreh_model = asreh_model_cpp.ASREHModel(
            in_channels=1,
            hct_dim=hct_dim,
            num_experts=num_experts
        )

        self.token_embedding = nn.Embedding(vocab_size, hct_dim)
        self.output_layer = nn.Linear(hct_dim, vocab_size)

    def forward(self,
                text_input: str,
                visual_input: Optional[torch.Tensor] = None,
                audio_input: Optional[torch.Tensor] = None,
                context: Optional[Dict] = None) -> Tuple[torch.Tensor, Any]:
        
        # New: Process all available inputs and get their conceptual vectors.
        text_vector = self.text_encoder(text_input, context=context)
        visual_vector = self.visual_encoder(visual_input) if visual_input is not None else None
        audio_vector = self.audio_encoder(audio_input, text_input) if audio_input is not None else None

        # New: Fuse the conceptual vectors into a single, multi-modal representation.
        all_vectors = [v for v in [text_vector, visual_vector, audio_vector] if v is not None]
        if not all_vectors:
            raise ValueError("At least one modality must be provided for a forward pass.")
            
        fused_representation = torch.mean(torch.cat(all_vectors, dim=0), dim=0).unsqueeze(0)
        
        # Use the C++ MoE router to select the conversational expert.
        conceptual_context = moe_router_cpp.ConceptualContext()
        conceptual_context.context_map = context if context else {'topic': ['conversation']}
        
        top_k_indices_np = self.cpp_moe_router.route(
            fused_representation.detach().cpu().numpy(),
            conceptual_context
        )
        top_k_indices = torch.from_numpy(top_k_indices_np).long()

        moe_output = fused_representation
        output_logits = self.output_layer(moe_output)

        decision_context = {
            'chosen_expert': top_k_indices[0].item(),
            'conceptual_factors': list(conceptual_context.context_map.keys())
        }

        return output_logits, decision_context
