# /src/modules/multi_modal_ckg_attention.py

"""
Multi-Modal CKG-Guided Attention with Cross-Modal Relationships
===============================================================
Extends CKG guidance to handle multi-modal inputs with cross-modal attention patterns.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from .ckg_guided_sparse_attention import CKGSparseAttention

class MultiModalCKGAttention(nn.Module):
    """
    Multi-modal attention with CKG-guided cross-modal relationships.
    """
    
    def __init__(self, 
                 dim: int,
                 num_heads: int,
                 ckg,
                 modalities: List[str] = ['text', 'visual', 'audio']):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.ckg = ckg
        self.modalities = modalities
        
        # Modality-specific attention layers
        self.modality_attentions = nn.ModuleDict({
            modality: CKGSparseAttention(dim, num_heads, ckg) 
            for modality in modalities
        })
        
        # Cross-modal attention
        self.cross_modal_attention = CKGSparseAttention(dim, num_heads, ckg)
        
        # Modality fusion with CKG guidance
        self.fusion_weights = nn.Parameter(torch.ones(len(modalities)))
        self.fusion_proj = nn.Linear(dim * len(modalities), dim)
        
        # Cross-modal relationship encoder
        self.cross_modal_relationship = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, num_heads),
            nn.Sigmoid()
        )
    
    def forward(self, 
                modality_embeddings: Dict[str, torch.Tensor],
                context: Dict = None) -> torch.Tensor:
        """
        Process multi-modal inputs with CKG-guided attention.
        """
        attended_modalities = {}
        cross_modal_guidance = {}
        
        # Intra-modal attention
        for modality, embedding in modality_embeddings.items():
            if modality in self.modality_attentions:
                mod_context = {**(context or {}), 'modality': modality}
                attended_output, _, guidance_info = self.modality_attentions[modality](
                    embedding, context=mod_context, return_attention_weights=True
                )
                attended_modalities[modality] = attended_output
                cross_modal_guidance[modality] = guidance_info
        
        # Cross-modal attention
        cross_modal_output = self._apply_cross_modal_attention(
            attended_modalities, context, cross_modal_guidance
        )
        
        # Fuse all modalities
        fused_output = self._fuse_modalities(attended_modalities, cross_modal_output)
        
        return fused_output
    
    def _apply_cross_modal_attention(self,
                                   attended_modalities: Dict[str, torch.Tensor],
                                   context: Dict,
                                   guidance_info: Dict) -> torch.Tensor:
        """
        Apply cross-modal attention using CKG relationships.
        """
        if len(attended_modalities) < 2:
            return list(attended_modalities.values())[0] if attended_modalities else None
        
        # Concatenate modality representations for cross-attention
        all_modality_embeddings = []
        modality_keys = []
        
        for modality, embedding in attended_modalities.items():
            all_modality_embeddings.append(embedding)
            modality_keys.append(modality)
        
        if not all_modality_embeddings:
            return None
        
        # Stack for cross-modal attention
        cross_modal_input = torch.cat(all_modality_embeddings, dim=1)
        
        # Apply cross-modal attention with CKG guidance
        cross_context = {**(context or {}), 'cross_modal': True, 'modalities': modality_keys}
        cross_output, _, _ = self.cross_modal_attention(
            cross_modal_input, context=cross_context, return_attention_weights=True
        )
        
        return cross_output
    
    def _fuse_modalities(self, 
                        modality_embeddings: Dict[str, torch.Tensor],
                        cross_modal_output: torch.Tensor) -> torch.Tensor:
        """
        Fuse modalities using CKG-guided weights.
        """
        modality_vectors = []
        
        # Add individual modality representations
        for modality, embedding in modality_embeddings.items():
            modality_vectors.append(embedding)
        
        # Add cross-modal representation if available
        if cross_modal_output is not None:
            # Split cross-modal output back to modalities
            split_size = cross_modal_output.shape[1] // len(modality_embeddings)
            for i in range(len(modality_embeddings)):
                start_idx = i * split_size
                end_idx = (i + 1) * split_size
                modality_vectors.append(cross_modal_output[:, start_idx:end_idx, :])
        
        # Weighted fusion
        weighted_vectors = []
        for i, vector in enumerate(modality_vectors):
            weight = self.fusion_weights[i % len(self.fusion_weights)]
            weighted_vectors.append(vector * weight)
        
        # Concatenate and project
        fused = torch.cat(weighted_vectors, dim=-1)
        return self.fusion_proj(fused)
    
    def get_cross_modal_report(self) -> Dict:
        """Get cross-modal attention performance report."""
        reports = {}
        for modality, attention_layer in self.modality_attentions.items():
            reports[modality] = attention_layer.get_ckg_performance_report()
        
        reports['cross_modal'] = self.cross_modal_attention.get_ckg_performance_report()
        
        return reports