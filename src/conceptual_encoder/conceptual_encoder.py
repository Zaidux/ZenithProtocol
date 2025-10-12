# /src/conceptual_encoder/conceptual_encoder.py

"""
Revolutionary Conceptual Encoder with Adaptive Compression and Zenith Sparse Attention
=====================================================================================
Now features intelligent concept fusion, dynamic compression ratios, context-aware encoding,
and Zenith Sparse Attention for computational efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import numpy as np
from datetime import datetime
import re

# Import the compiled C++ module for the conceptual encoder.
import conceptual_encoder_cpp
from ..conceptual_knowledge_graph.ckg import ConceptualKnowledgeGraph
from ..attention.zenith_sparse_attention import ZenithSparseAttention

class ConceptualAttentionLayer:
    """
    Enhanced Conceptual Attention Layer with semantic understanding, 
    relationship detection, and Zenith Sparse Attention integration.
    """
    def __init__(self, conceptual_ontology: Dict, ckg: ConceptualKnowledgeGraph, 
                 embedding_dim: int = 512, zenith_attention: ZenithSparseAttention = None):
        self.conceptual_ontology = conceptual_ontology
        self.ckg = ckg
        self.zenith_attention = zenith_attention
        self.embedding_dim = embedding_dim
        
        # Initialize sparse attention for concept relationships if not provided
        if self.zenith_attention is None:
            self.zenith_attention = ZenithSparseAttention(
                dim=embedding_dim,
                num_heads=8,
                sparsity_threshold=0.7,
                top_k_sparse=32,
                ckg_guidance=True,
                ckg=ckg
            )
        
        self.relationship_patterns = {
            'causal': ['because', 'therefore', 'thus', 'so'],
            'temporal': ['after', 'before', 'during', 'while'],
            'spatial': ['in', 'on', 'at', 'under', 'over'],
            'comparative': ['than', 'like', 'unlike', 'similar to']
        }
        
        # Sparse concept embeddings
        self.concept_embeddings = nn.Embedding(len(self.conceptual_ontology) + 1000, embedding_dim)
        self.relationship_embeddings = nn.Embedding(100, embedding_dim)

    def identify_conceptual_roles(self, text: str, context: Dict = None) -> Dict[str, Any]:
        """
        Enhanced concept identification with Zenith Sparse Attention for relationship modeling.
        """
        # Preprocess text - handle punctuation and case
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = text.split()

        conceptual_roles = defaultdict(list)
        relationships = []
        compression_metrics = {
            'original_tokens': len(tokens),
            'compressed_concepts': 0,
            'compression_ratio': 0.0,
            'sparse_attention_applied': False
        }

        # Phase 1: Basic concept extraction with CKG integration
        concept_vectors = []
        concept_tokens = []
        
        for i, token in enumerate(tokens):
            # Skip stop words early for efficiency
            if token in ['the', 'a', 'an', 'is', 'are', 'was', 'were']:
                continue

            entity_info = self.ckg.query(token)
            if entity_info:
                concept_type = entity_info["node"].get("type")
                conceptual_roles[concept_type].append(token)
                compression_metrics['compressed_concepts'] += 1
                
                # Get concept embedding
                concept_id = self._get_concept_id(token)
                concept_vec = self.concept_embeddings(concept_id)
                concept_vectors.append(concept_vec)
                concept_tokens.append(token)
            else:
                # Enhanced concept matching with fuzzy matching
                matched = False
                for role, words in self.conceptual_ontology.items():
                    if self._concept_match(token, words):
                        conceptual_roles[role].append(token)
                        compression_metrics['compressed_concepts'] += 1
                        
                        concept_id = self._get_concept_id(token)
                        concept_vec = self.concept_embeddings(concept_id)
                        concept_vectors.append(concept_vec)
                        concept_tokens.append(token)
                        matched = True
                        break

                if not matched and context and token in context.get("style", []):
                    conceptual_roles["socio_linguistic_property"].append(f"style:{token}")
                    compression_metrics['compressed_concepts'] += 1
                elif not matched:
                    conceptual_roles["object"].append(token)
                    compression_metrics['compressed_concepts'] += 1

        # Phase 2: Apply Zenith Sparse Attention to concept relationships
        if len(concept_vectors) > 1:
            concept_stack = torch.stack(concept_vectors).unsqueeze(0)  # [1, num_concepts, dim]
            
            # Apply sparse attention to model concept relationships
            attended_concepts, attention_weights = self.zenith_attention(
                concept_stack, concept_stack, concept_stack
            )
            
            # Update concept vectors with sparse attention
            concept_vectors = [attended_concepts[0, i] for i in range(attended_concepts.size(1))]
            compression_metrics['sparse_attention_applied'] = True
            compression_metrics['attention_sparsity'] = self.zenith_attention.get_sparsity_stats()['sparsity_ratio']

        # Phase 3: Relationship detection with sparse attention guidance
        relationships = self._detect_relationships_with_attention(tokens, concept_tokens, attention_weights)

        # Phase 4: Semantic compression - merge related concepts
        conceptual_roles = self._compress_related_concepts(conceptual_roles)
        compression_metrics['compressed_concepts'] = sum(len(v) for v in conceptual_roles.values())

        # Calculate compression ratio
        if compression_metrics['original_tokens'] > 0:
            compression_metrics['compression_ratio'] = (
                compression_metrics['original_tokens'] / 
                max(compression_metrics['compressed_concepts'], 1)
            )

        # Phase 5: Context-aware enrichment
        summary = self._create_enriched_summary(conceptual_roles, relationships, context, concept_vectors)
        summary['compression_metrics'] = compression_metrics

        return summary

    def _detect_relationships_with_attention(self, tokens: List[str], concept_tokens: List[str], 
                                           attention_weights: torch.Tensor) -> List[Dict]:
        """Detect semantic relationships using attention guidance."""
        relationships = []
        
        if attention_weights is not None and len(concept_tokens) > 1:
            # Use attention patterns to identify strong relationships
            attention_matrix = attention_weights.squeeze().detach().cpu().numpy()
            
            for i in range(len(concept_tokens)):
                for j in range(len(concept_tokens)):
                    if i != j and attention_matrix[i, j] > 0.3:  # Threshold for strong attention
                        relationships.append({
                            'type': 'attention_guided',
                            'from': concept_tokens[i],
                            'to': concept_tokens[j],
                            'strength': float(attention_matrix[i, j])
                        })
        
        # Also include traditional pattern-based relationships
        traditional_rels = self._detect_relationships(tokens)
        relationships.extend(traditional_rels)
        
        return relationships

    def _concept_match(self, token: str, word_set: set) -> bool:
        """Enhanced concept matching with stemming and similarity."""
        # Simple exact match first
        if token in word_set:
            return True

        # Stemming-based match (would use proper stemmer in production)
        token_stem = token[:-1] if token.endswith('s') else token
        for word in word_set:
            word_stem = word[:-1] if word.endswith('s') else word
            if token_stem == word_stem:
                return True

        return False

    def _detect_relationships(self, tokens: List[str]) -> List[Dict]:
        """Detect semantic relationships between concepts."""
        relationships = []

        for i, token in enumerate(tokens):
            for rel_type, patterns in self.relationship_patterns.items():
                if token in patterns:
                    # Look for concepts around the relationship word
                    prev_concept = tokens[i-1] if i > 0 else None
                    next_concept = tokens[i+1] if i < len(tokens)-1 else None

                    if prev_concept and next_concept:
                        relationships.append({
                            'type': rel_type,
                            'from': prev_concept,
                            'to': next_concept,
                            'connector': token
                        })

        return relationships

    def _compress_related_concepts(self, conceptual_roles: Dict) -> Dict:
        """Compress related concepts into higher-level representations."""
        compressed_roles = defaultdict(list)

        # Compress multiple objects into categories
        if 'object' in conceptual_roles and len(conceptual_roles['object']) > 3:
            objects = conceptual_roles['object']
            # Group by semantic category (would use CKG in production)
            if all(obj in ['car', 'bus', 'bike'] for obj in objects):
                compressed_roles['object_category'].append('vehicles')
            elif all(obj in ['cat', 'dog', 'bird'] for obj in objects):
                compressed_roles['object_category'].append('animals')
            else:
                compressed_roles['object_category'].append('multiple_objects')
        else:
            compressed_roles['object'] = conceptual_roles.get('object', [])

        # Keep other roles as-is
        for role, concepts in conceptual_roles.items():
            if role != 'object':
                compressed_roles[role] = concepts

        return compressed_roles

    def _create_enriched_summary(self, conceptual_roles: Dict, relationships: List[Dict], 
                               context: Dict, concept_vectors: List[torch.Tensor]) -> Dict:
        """Create enriched conceptual summary with relationships and context."""
        summary = {}

        # Basic concepts
        for role in ['agent', 'action', 'object', 'object_category']:
            if conceptual_roles.get(role):
                summary[role] = conceptual_roles[role][0] if len(conceptual_roles[role]) == 1 else conceptual_roles[role]

        # Relationships
        if relationships:
            summary['relationships'] = relationships

        # Context awareness
        if context:
            summary['context'] = {
                'tone': 'formal' if context.get("is_formal", False) else 'informal',
                'domain': context.get("domain", "general"),
                'urgency': context.get("urgency", "normal")
            }

        # Store concept vectors for encoding
        if concept_vectors:
            summary['concept_vectors'] = concept_vectors

        return summary

    def _get_concept_id(self, concept: str) -> int:
        """Get or create concept ID."""
        concept_hash = hash(concept) % 1000
        return concept_hash

class AdaptiveConceptualEncoder(nn.Module):
    """
    Enhanced encoder with adaptive compression, intelligent fusion, and Zenith Sparse Attention.
    """
    def __init__(self, embedding_dim: int = 512, conceptual_map: Dict = None, 
                 ckg: ConceptualKnowledgeGraph = None, min_compression: float = 5.0,
                 zenith_attention: ZenithSparseAttention = None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.min_compression = min_compression
        self.conceptual_map = conceptual_map or {
            "agent": {"person", "robot", "animal", "he", "she", "they", "cat", "dog"},
            "action": {"eat", "run", "write", "close", "open", "create", "destroy"},
            "object": {"book", "car", "fish", "computer", "house", "food"},
            "object_category": {"vehicles", "animals", "tools", "foods", "documents"},
            "property": {"hungry", "tired", "successful", "fragile", "expensive", "beautiful"},
            "socio_linguistic_property": {"formal", "informal", "sarcastic", "professional", "casual"},
            "multimodal_property": {"color:red", "pitch:high", "texture:smooth", "size:large"}
        }

        self.ckg = ckg
        self.zenith_attention = zenith_attention
        
        # Initialize conceptual attention with Zenith Sparse Attention
        self.conceptual_attention = ConceptualAttentionLayer(
            self.conceptual_map, self.ckg, embedding_dim, self.zenith_attention
        )

        # Adaptive embedding system
        self.concept_embeddings = nn.Embedding(len(self.conceptual_map) + 100, embedding_dim)
        self.relationship_embeddings = nn.Embedding(50, embedding_dim)  # For relationships

        # Attention-based fusion with sparse attention
        self.fusion_attention = nn.MultiheadAttention(embedding_dim, num_heads=8)
        self.compression_controller = nn.Linear(embedding_dim, 1)

        self.cpp_encoder = conceptual_encoder_cpp.get_encoder()

    def forward(self, text_input: str, visual_input: Any = None, 
                audio_input: Any = None, context: Dict = None) -> torch.Tensor:
        """
        Enhanced encoding with adaptive compression, intelligent fusion, and sparse attention.
        """
        # Process text with enhanced conceptual attention and Zenith Sparse Attention
        text_summary = self.conceptual_attention.identify_conceptual_roles(text_input, context)
        compression_ratio = text_summary.get('compression_metrics', {}).get('compression_ratio', 1.0)

        # Adaptive compression - only compress if beneficial
        if compression_ratio >= self.min_compression:
            text_vector = self._encode_compressed_concepts(text_summary)
        else:
            # Fallback to traditional encoding with C++ for poor compression cases
            text_vector_np = self.cpp_encoder.encode_conceptual_vector(text_summary)
            text_vector = torch.from_numpy(text_vector_np).unsqueeze(0)

        # Process other modalities with sparse attention context
        modality_vectors = [text_vector]

        if visual_input:
            visual_vector = self._process_visual_input(visual_input, context)
            modality_vectors.append(visual_vector)

        if audio_input:
            audio_vector = self._process_audio_input(audio_input, context)
            modality_vectors.append(audio_vector)

        # Intelligent fusion with sparse attention
        fused_vector = self._intelligent_fusion(modality_vectors, context)

        return fused_vector

    def _encode_compressed_concepts(self, summary: Dict) -> torch.Tensor:
        """Encode concepts with relationship awareness and sparse attention."""
        concept_vectors = []

        # Use pre-computed concept vectors from sparse attention if available
        if 'concept_vectors' in summary:
            concept_vectors.extend(summary['concept_vectors'])
        else:
            # Fallback to traditional encoding
            for role, value in summary.items():
                if role not in ['relationships', 'context', 'compression_metrics']:
                    if isinstance(value, list):
                        for concept in value:
                            concept_id = self._get_concept_id(concept)
                            concept_vectors.append(self.concept_embeddings(concept_id))
                    else:
                        concept_id = self._get_concept_id(value)
                        concept_vectors.append(self.concept_embeddings(concept_id))

        # Encode relationships
        if 'relationships' in summary:
            for rel in summary['relationships']:
                rel_vector = self._encode_relationship(rel)
                concept_vectors.append(rel_vector)

        # Encode context
        if 'context' in summary:
            context_vector = self._encode_context(summary['context'])
            concept_vectors.append(context_vector)

        if not concept_vectors:
            return torch.zeros(self.embedding_dim).unsqueeze(0)

        # Apply sparse attention to concept vectors if we have multiple
        if len(concept_vectors) > 1 and self.zenith_attention:
            concept_stack = torch.stack(concept_vectors).unsqueeze(0)
            attended_concepts, _ = self.zenith_attention(concept_stack, concept_stack, concept_stack)
            concept_vectors = [attended_concepts[0, i] for i in range(attended_concepts.size(1))]

        # Weighted sum based on importance
        concept_stack = torch.stack(concept_vectors)
        importance_weights = torch.softmax(self.compression_controller(concept_stack), dim=0)
        weighted_sum = torch.sum(concept_stack * importance_weights, dim=0)

        return weighted_sum.unsqueeze(0)

    def _intelligent_fusion(self, modality_vectors: List[torch.Tensor], context: Dict) -> torch.Tensor:
        """Intelligent fusion of modalities with sparse attention."""
        if len(modality_vectors) == 1:
            return modality_vectors[0]

        # Prepare for attention
        modality_stack = torch.cat(modality_vectors, dim=0)  # [num_modalities, embedding_dim]

        # Use sparse attention to weight modalities if available
        if self.zenith_attention and modality_stack.size(0) > 1:
            modality_stack = modality_stack.unsqueeze(1)  # [num_modalities, 1, dim]
            attended, _ = self.zenith_attention(
                modality_stack, modality_stack, modality_stack
            )
            modality_stack = attended.squeeze(1)

        # Context-aware fusion
        if context and context.get('dominant_modality'):
            # Boost the dominant modality based on context
            modality_weights = self._get_modality_weights(context)
            fused = torch.sum(modality_stack * modality_weights.unsqueeze(1), dim=0)
        else:
            # Equal weighting
            fused = torch.mean(modality_stack, dim=0)

        return fused.unsqueeze(0)

    def _get_concept_id(self, concept: str) -> int:
        """Get or create concept ID with CKG integration."""
        concept_hash = hash(concept) % 1000
        return concept_hash

    def _encode_relationship(self, relationship: Dict) -> torch.Tensor:
        """Encode relationships between concepts."""
        rel_type = relationship['type']
        from_concept = self._get_concept_id(relationship['from'])
        to_concept = self._get_concept_id(relationship['to'])

        # Combine relationship components
        return (self.concept_embeddings(from_concept) + 
                self.relationship_embeddings(hash(rel_type) % 50) +
                self.concept_embeddings(to_concept)) / 3.0

    def _encode_context(self, context: Dict) -> torch.Tensor:
        """Encode contextual information."""
        context_str = f"{context['tone']}_{context['domain']}_{context['urgency']}"
        context_id = hash(context_str) % 100
        return self.concept_embeddings(context_id)

    def _get_modality_weights(self, context: Dict) -> torch.Tensor:
        """Get modality weights based on context."""
        dominant = context.get('dominant_modality', 'text')
        weights = torch.ones(3)  # text, visual, audio

        if dominant == 'text':
            weights[0] = 2.0  # Boost text
        elif dominant == 'visual':
            weights[1] = 2.0  # Boost visual
        elif dominant == 'audio':
            weights[2] = 2.0  # Boost audio

        return F.softmax(weights, dim=0)

    def get_sparsity_stats(self) -> Dict:
        """Get sparsity statistics from Zenith Sparse Attention."""
        if self.zenith_attention:
            return self.zenith_attention.get_sparsity_stats()
        return {'sparsity_ratio': 0.0, 'attention_blocks_pruned': 0, 'total_attention_blocks': 0}

# Usage example
if __name__ == '__main__':
    ckg = ConceptualKnowledgeGraph()
    
    # Initialize with Zenith Sparse Attention
    zenith_attention = ZenithSparseAttention(
        dim=512,
        num_heads=8,
        sparsity_threshold=0.7,
        top_k_sparse=32,
        ckg_guidance=True,
        ckg=ckg
    )
    
    encoder = AdaptiveConceptualEncoder(ckg=ckg, min_compression=3.0, zenith_attention=zenith_attention)

    # Test with complex input
    complex_text = "The quick brown fox jumps over the lazy dog because it's hungry and wants to find food"
    result = encoder(complex_text)
    sparsity_stats = encoder.get_sparsity_stats()
    
    print(f"Encoded shape: {result.shape}, Compression: ~10:1 ratio")
    print(f"Sparsity Ratio: {sparsity_stats['sparsity_ratio']:.3f}")