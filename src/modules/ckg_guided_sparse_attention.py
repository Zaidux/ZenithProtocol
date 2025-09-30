# /src/modules/ckg_guided_sparse_attention.py

"""
CKG-Guided Sparse Attention with Dynamic Pattern Adaptation
===========================================================
Uses the Conceptual Knowledge Graph to intelligently determine which token pairs
should attend to each other based on semantic relationships and causal rules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np
from collections import defaultdict
import json

from .conceptual_sparse_attention import ConceptualSparseAttention
from ..conceptual_knowledge_graph.ckg import ConceptualKnowledgeGraph
from ..conceptual_encoder.conceptual_encoder import AdaptiveConceptualEncoder

class CKGSparseAttention(ConceptualSparseAttention):
    """
    Enhanced sparse attention that uses CKG for intelligent pattern selection.
    Learns optimal attention patterns based on conceptual relationships.
    """
    
    def __init__(self, 
                 dim: int,
                 num_heads: int = 8,
                 ckg: ConceptualKnowledgeGraph = None,
                 sparsity_ratio: float = 0.15,
                 max_seq_length: int = 4096,
                 pattern_memory_size: int = 1000,
                 adaptive_learning_rate: float = 0.01):
        super().__init__(dim, num_heads, ckg, sparsity_ratio, max_seq_length)
        
        self.pattern_memory_size = pattern_memory_size
        self.adaptive_learning_rate = adaptive_learning_rate
        
        # CKG-guided attention controllers
        self.relationship_attention = nn.Sequential(
            nn.Linear(dim * 2, dim),  # Pairwise relationship encoding
            nn.ReLU(),
            nn.Linear(dim, num_heads),
            nn.Sigmoid()
        )
        
        # Causal rule applier
        self.rule_compliance_scorer = nn.Sequential(
            nn.Linear(dim * 3, dim),  # source, target, relationship
            nn.ReLU(), 
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        
        # Pattern memory bank (learns optimal attention patterns)
        self.pattern_memory = nn.ParameterDict({
            'domain_patterns': nn.Parameter(torch.randn(10, max_seq_length, max_seq_length) * 0.1),
            'concept_patterns': nn.Parameter(torch.randn(50, max_seq_length // 4, max_seq_length // 4) * 0.1),
            'relationship_patterns': nn.Parameter(torch.randn(20, num_heads, 32, 32) * 0.1)
        })
        
        # Attention pattern cache
        self.attention_pattern_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # CKG integration metrics
        self.ckg_usage_stats = {
            'rule_checks': 0,
            'relationship_uses': 0,
            'causal_violations_prevented': 0,
            'conceptual_matches_found': 0
        }
    
    def forward(self, 
                x: torch.Tensor,
                context: Dict = None,
                return_attention_weights: bool = False,
                use_ckg_guidance: bool = True) -> torch.Tensor:
        """
        Forward pass with CKG-guided sparse attention.
        """
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Generate CKG-guided sparse attention mask
        if use_ckg_guidance and seq_len > 16:  # Use CKG for non-trivial sequences
            attention_mask, guidance_info = self._generate_ckg_guided_mask(
                x, seq_len, context
            )
        else:
            attention_mask = self._generate_conceptual_sparse_mask(x, seq_len, context)
            guidance_info = {}
        
        # Compute sparse attention
        attn_output, attn_weights = self._sparse_attention(
            q, k, v, attention_mask, seq_len
        )
        
        # Learn from this attention pattern if guidance was used
        if use_ckg_guidance and guidance_info:
            self._update_pattern_memory(attn_weights, guidance_info, seq_len)
        
        # Combine heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.dim
        )
        output = self.out_proj(attn_output)
        
        # Update statistics
        self._update_attention_stats(seq_len, attention_mask)
        self._update_ckg_stats(guidance_info)
        
        if return_attention_weights:
            return output, attn_weights, guidance_info
        return output
    
    def _generate_ckg_guided_mask(self, 
                                x: torch.Tensor,
                                seq_len: int,
                                context: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Generate attention mask using CKG guidance.
        """
        batch_size = x.shape[0]
        guidance_info = {
            'rules_applied': [],
            'relationships_used': [],
            'conceptual_matches': [],
            'cache_used': False
        }
        
        # Try cache first
        cache_key = self._generate_cache_key(x, context, seq_len)
        if cache_key in self.attention_pattern_cache:
            self.cache_hits += 1
            guidance_info['cache_used'] = True
            return self.attention_pattern_cache[cache_key], guidance_info
        
        self.cache_misses += 1
        
        # Initialize mask with local attention baseline
        base_mask = self._create_local_attention_mask(seq_len, batch_size, x.device)
        
        # Apply CKG-guided enhancements
        enhanced_mask = self._apply_ckg_enhancements(
            base_mask, x, seq_len, context, guidance_info
        )
        
        # Apply causal rules and constraints
        final_mask = self._apply_causal_constraints(
            enhanced_mask, x, seq_len, context, guidance_info
        )
        
        # Cache the pattern
        if len(self.attention_pattern_cache) < self.pattern_memory_size:
            self.attention_pattern_cache[cache_key] = final_mask
        
        return final_mask, guidance_info
    
    def _apply_ckg_enhancements(self,
                              base_mask: torch.Tensor,
                              x: torch.Tensor,
                              seq_len: int,
                              context: Dict,
                              guidance_info: Dict) -> torch.Tensor:
        """
        Apply CKG-based enhancements to attention mask.
        """
        enhanced_mask = base_mask.clone()
        batch_size = x.shape[0]
        
        # Extract conceptual features from input
        conceptual_features = self._extract_conceptual_features(x, context)
        
        # Apply relationship-based attention
        relationship_mask = self._apply_relationship_attention(
            conceptual_features, seq_len, x.device
        )
        enhanced_mask = torch.max(enhanced_mask, relationship_mask)
        guidance_info['relationships_used'] = list(relationship_mask.unique().cpu().numpy())
        
        # Apply domain-specific patterns
        if context and 'domain' in context:
            domain_mask = self._get_domain_pattern(context['domain'], seq_len, batch_size, x.device)
            enhanced_mask = torch.max(enhanced_mask, domain_mask)
            guidance_info['rules_applied'].append(f"domain_{context['domain']}")
        
        # Apply conceptual similarity attention
        similarity_mask = self._apply_conceptual_similarity(
            conceptual_features, seq_len, x.device
        )
        enhanced_mask = torch.max(enhanced_mask, similarity_mask)
        guidance_info['conceptual_matches'] = [f"similarity_{similarity_mask.sum().item()}"]
        
        return enhanced_mask
    
    def _apply_relationship_attention(self,
                                    conceptual_features: Dict,
                                    seq_len: int,
                                    device: torch.device) -> torch.Tensor:
        """
        Apply attention based on CKG relationships.
        """
        batch_size = 1  # Simplified for this example
        mask = torch.zeros(batch_size, self.num_heads, seq_len, seq_len, device=device)
        
        if not self.ckg or 'concepts' not in conceptual_features:
            return mask
        
        # Query CKG for relationships between concepts
        concepts = conceptual_features['concepts']
        for i, source_concept in enumerate(concepts[:seq_len]):
            for j, target_concept in enumerate(concepts[:seq_len]):
                if i == j:
                    continue
                
                # Check if concepts are related in CKG
                relationship_strength = self._get_relationship_strength(
                    source_concept, target_concept
                )
                
                if relationship_strength > 0.3:  # Threshold for relationship attention
                    # Apply to all heads or distribute based on relationship type
                    for head in range(self.num_heads):
                        mask[0, head, i, j] = 1.0
                    
                    self.ckg_usage_stats['relationship_uses'] += 1
        
        return mask
    
    def _get_relationship_strength(self, source: str, target: str) -> float:
        """
        Get relationship strength between two concepts from CKG.
        """
        try:
            # Query CKG for relationships
            source_data = self.ckg.query(source)
            target_data = self.ckg.query(target)
            
            if not source_data or not target_data:
                return 0.0
            
            # Check for direct relationships
            for connection in source_data.get('connections', []):
                if connection.get('target_id') == target:
                    return 0.8  # Strong relationship
            
            # Check for shared properties or causal rules
            source_props = set(source_data.get('node', {}).get('properties', {}).keys())
            target_props = set(target_data.get('node', {}).get('properties', {}).keys())
            
            shared_props = source_props.intersection(target_props)
            if shared_props:
                return 0.5  # Moderate relationship
            
            # Check causal rules
            for rule_id, rule in self.ckg.causal_rules.items():
                conditions = ' '.join(rule.get('conditions', [])).lower()
                effects = ' '.join(rule.get('effects', [])).lower()
                
                if source.lower() in conditions and target.lower() in effects:
                    return 0.7  # Causal relationship
            
            return 0.0
            
        except Exception as e:
            print(f"Error checking relationship strength: {e}")
            return 0.0
    
    def _apply_conceptual_similarity(self,
                                  conceptual_features: Dict,
                                  seq_len: int,
                                  device: torch.device) -> torch.Tensor:
        """
        Apply attention based on conceptual similarity.
        """
        batch_size = 1
        mask = torch.zeros(batch_size, self.num_heads, seq_len, seq_len, device=device)
        
        if 'embeddings' not in conceptual_features:
            return mask
        
        embeddings = conceptual_features['embeddings']
        if len(embeddings) < 2:
            return mask
        
        # Calculate cosine similarity between concept embeddings
        similarity_matrix = F.cosine_similarity(
            embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1
        )
        
        # Apply similarity threshold
        similarity_threshold = 0.6
        similar_pairs = (similarity_matrix > similarity_threshold).float()
        
        # Expand to full attention mask
        for head in range(self.num_heads):
            mask[0, head, :len(embeddings), :len(embeddings)] = similar_pairs
        
        self.ckg_usage_stats['conceptual_matches_found'] += similar_pairs.sum().item()
        
        return mask
    
    def _apply_causal_constraints(self,
                               mask: torch.Tensor,
                               x: torch.Tensor,
                               seq_len: int,
                               context: Dict,
                               guidance_info: Dict) -> torch.Tensor:
        """
        Apply causal constraints to prevent attention violations.
        """
        constrained_mask = mask.clone()
        
        if not context or 'domain' not in context:
            return constrained_mask
        
        domain = context['domain']
        
        # Apply domain-specific causal constraints
        for rule_id, rule in self.ckg.causal_rules.items():
            if rule.get('domain') == domain or rule.get('domain') is None:
                violations_prevented = self._apply_causal_rule(
                    constrained_mask, rule, seq_len, x.device
                )
                if violations_prevented:
                    guidance_info['rules_applied'].append(rule_id)
                    self.ckg_usage_stats['causal_violations_prevented'] += violations_prevented
        
        return constrained_mask
    
    def _apply_causal_rule(self,
                         mask: torch.Tensor,
                         rule: Dict,
                         seq_len: int,
                         device: torch.device) -> int:
        """
        Apply a single causal rule to the attention mask.
        Returns number of violations prevented.
        """
        violations_prevented = 0
        
        # Simplified rule application - in practice would parse rule conditions/effects
        rule_type = rule.get('type', 'generic')
        
        if 'prevent' in rule.get('description', '').lower():
            # This is a prevention rule - block certain attention patterns
            conditions = rule.get('conditions', [])
            effects = rule.get('effects', [])
            
            # Simple keyword-based rule application
            for condition in conditions:
                if 'gap' in condition.lower() and 'create' in condition.lower():
                    # Prevent attention that might lead to gap creation
                    # This is domain-specific logic
                    pass
        
        return violations_prevented
    
    def _extract_conceptual_features(self, x: torch.Tensor, context: Dict) -> Dict:
        """
        Extract conceptual features from input using conceptual encoder.
        """
        try:
            # Use the conceptual encoder to get concept information
            if hasattr(self, 'conceptual_encoder') and context:
                # For text sequences, we'd need to convert back to text
                # This is simplified - in practice would use proper text reconstruction
                conceptual_summary = {
                    'concepts': [f"token_{i}" for i in range(min(10, x.shape[1]))],
                    'embeddings': x[:, :10, :].mean(dim=1)  # Simplified embedding
                }
                return conceptual_summary
        except Exception as e:
            print(f"Conceptual feature extraction failed: {e}")
        
        return {'concepts': [], 'embeddings': torch.tensor([])}
    
    def _create_local_attention_mask(self, seq_len: int, batch_size: int, device: torch.device) -> torch.Tensor:
        """Create baseline local attention mask."""
        mask = torch.zeros(batch_size, self.num_heads, seq_len, seq_len, device=device)
        
        # Local window attention
        window_size = min(64, seq_len)
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            mask[:, :, i, start:end] = 1.0
        
        return mask
    
    def _get_domain_pattern(self, domain: str, seq_len: int, batch_size: int, device: torch.device) -> torch.Tensor:
        """Get domain-specific attention pattern."""
        # Simplified domain patterns
        pattern = torch.zeros(batch_size, self.num_heads, seq_len, seq_len, device=device)
        
        if domain == 'tetris':
            # Tetris: focus on spatial relationships and future consequences
            for i in range(seq_len):
                # Attend to immediate neighbors and strategic positions
                pattern[:, :, i, max(0, i-2):min(seq_len, i+3)] = 1.0
                # Global attention to key positions (simplified)
                if i % 5 == 0:  # Key positions
                    pattern[:, :, i, :] = 1.0
                    
        elif domain == 'chess':
            # Chess: strategic long-range dependencies
            for i in range(seq_len):
                # Local context
                pattern[:, :, i, max(0, i-3):min(seq_len, i+4)] = 1.0
                # Strategic long-range connections
                if i % 8 == 0:
                    pattern[:, :, i, ::8] = 1.0  # Regular strategic points
        
        return pattern
    
    def _generate_cache_key(self, x: torch.Tensor, context: Dict, seq_len: int) -> str:
        """Generate cache key for attention patterns."""
        domain = context.get('domain', 'unknown') if context else 'unknown'
        content_hash = hash(str(x[:, :min(10, seq_len), :].detach().cpu().numpy().tobytes()))
        return f"{domain}_{seq_len}_{content_hash}"
    
    def _update_pattern_memory(self, attn_weights: torch.Tensor, guidance_info: Dict, seq_len: int):
        """Learn from successful attention patterns."""
        # Simplified pattern learning - would be more sophisticated
        if guidance_info.get('rules_applied') and attn_weights.sum() > 0:
            # Positive reinforcement for rules that worked well
            pass
    
    def _update_ckg_stats(self, guidance_info: Dict):
        """Update CKG usage statistics."""
        self.ckg_usage_stats['rule_checks'] += len(guidance_info.get('rules_applied', []))
    
    def get_ckg_performance_report(self) -> Dict:
        """Get CKG integration performance report."""
        cache_hit_rate = self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
        
        return {
            **self.get_performance_report(),
            'ckg_cache_hit_rate': f"{cache_hit_rate:.3f}",
            'relationship_uses': self.ckg_usage_stats['relationship_uses'],
            'causal_violations_prevented': self.ckg_usage_stats['causal_violations_prevented'],
            'conceptual_matches_found': self.ckg_usage_stats['conceptual_matches_found'],
            'rule_checks_performed': self.ckg_usage_stats['rule_checks']
        }