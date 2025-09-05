# /src/conceptual_encoder/conceptual_encoder.py

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import numpy as np
from datetime import datetime

# Import the compiled C++ module for the conceptual encoder.
import conceptual_encoder_cpp
from ..conceptual_knowledge_graph.ckg import ConceptualKnowledgeGraph # New: Import CKG

class ConceptualAttentionLayer:
    """
    The Conceptual Attention Layer now goes beyond simple roles and considers socio-linguistic
    and contextual cues to enrich the conceptual understanding.
    """
    def __init__(self, conceptual_ontology: Dict, ckg: ConceptualKnowledgeGraph):
        self.conceptual_ontology = conceptual_ontology
        self.ckg = ckg

    def identify_conceptual_roles(self, text: str, context: Dict = None) -> Dict[str, Any]:
        """
        Identifies conceptual roles and infers socio-linguistic properties from text and context.
        """
        tokens = text.lower().split()
        conceptual_roles = defaultdict(list)
        
        # New: Use the CKG to look up pre-defined properties and relationships.
        for token in tokens:
            entity_info = self.ckg.query(token)
            if entity_info:
                # Assuming the CKG query returns conceptual information.
                conceptual_roles["ckg_property"].append(entity_info["node"].get("type"))
            else:
                for role, words in self.conceptual_ontology.items():
                    if token in words:
                        conceptual_roles[role].append(token)
                        break
                else: # New: If no role is found, check for socio-linguistic properties.
                    if context and token in context.get("style", []):
                        conceptual_roles["socio_linguistic_property"].append(f"style:{token}")
                    else:
                        conceptual_roles["object"].append(token)
        
        # New: Infer socio-linguistic properties based on the prompt's context.
        if context:
            if context.get("is_formal", False):
                conceptual_roles["tone"].append("formal")
            else:
                conceptual_roles["tone"].append("informal")
        
        # Re-apply the "Connect" and "Assume" steps with enriched information
        summary = {}
        if "agent" in conceptual_roles:
            summary["agent"] = conceptual_roles["agent"][0]
        if "action" in conceptual_roles:
            summary["action"] = conceptual_roles["action"][0]
        if "object" in conceptual_roles:
            summary["object"] = conceptual_roles["object"]
        if "tone" in conceptual_roles:
            summary["tone"] = conceptual_roles["tone"][0]

        return summary

class ZenithConceptualEncoder(nn.Module):
    """
    The Zenith Conceptual Encoder is now multi-modal, capable of processing
    text, visual, and audio inputs into a unified conceptual vector.
    """
    def __init__(self, embedding_dim: int = 512, conceptual_map: Dict = None, ckg: ConceptualKnowledgeGraph = None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.conceptual_map = conceptual_map or {
            "agent": {"person", "robot", "animal", "he", "cat"},
            "action": {"eat", "run", "write", "close", "ate"},
            "object": {"book", "car", "fish"},
            "motion": {"writing", "running"},
            "bridge": {"because", "the", "in", "after"},
            "property": {"hungry", "tired", "successful", "fragile"},
            # New: Add socio-linguistic properties
            "socio_linguistic_property": {"formal", "informal", "sarcastic"},
            # New: Add multimodal properties
            "multimodal_property": {"color:red", "pitch:high"}
        }
        
        # New: Initialize a CKG instance
        self.ckg = ckg
        # New: Pass the CKG to the Conceptual Attention Layer
        self.conceptual_attention_layer = ConceptualAttentionLayer(self.conceptual_map, self.ckg)

        all_concepts = list(self.conceptual_map.keys())
        all_props = set(item for sublist in self.conceptual_map.values() for item in sublist)
        self.concept_to_id = {concept: i for i, concept in enumerate(all_concepts)}
        self.prop_to_id = {prop: i for i, prop in enumerate(all_props)}
        self.embedding_layer = nn.Embedding(len(self.concept_to_id) + len(self.prop_to_id) + 1, embedding_dim)
        
        # Initialize the C++ encoder instance.
        self.cpp_encoder = conceptual_encoder_cpp.get_encoder()

    def forward(self, text_input: str, visual_input: Any = None, audio_input: Any = None, context: Dict = None) -> torch.Tensor:
        """
        Encodes all modalities into a single, fused conceptual vector.
        """
        conceptual_vectors = []
        
        # Process text input
        if text_input:
            text_summary = self.conceptual_attention_layer.identify_conceptual_roles(text_input, context)
            text_vector_np = self.cpp_encoder.encode_conceptual_vector(text_summary)
            conceptual_vectors.append(torch.from_numpy(text_vector_np).unsqueeze(0))
            
        # New: Process visual and audio inputs (conceptual placeholders)
        if visual_input:
            visual_summary = self._process_visual_input(visual_input)
            visual_vector_np = self.cpp_encoder.encode_conceptual_vector(visual_summary)
            conceptual_vectors.append(torch.from_numpy(visual_vector_np).unsqueeze(0))

        if audio_input:
            audio_summary = self._process_audio_input(audio_input)
            audio_vector_np = self.cpp_encoder.encode_conceptual_vector(audio_summary)
            conceptual_vectors.append(torch.from_numpy(audio_vector_np).unsqueeze(0))
            
        # New: Fuse all conceptual vectors into one.
        if not conceptual_vectors:
            return torch.zeros(self.embedding_dim).unsqueeze(0)
            
        fused_vector = torch.mean(torch.cat(conceptual_vectors, dim=0), dim=0).unsqueeze(0)
        return fused_vector

    def _process_visual_input(self, visual_input: Any) -> Dict[str, Any]:
        """Conceptual placeholder for the Visual Encoder."""
        # This would be where conceptual_visual_encoder.py logic resides.
        return {"agent": "person", "object": "car", "multimodal_property": "color:red"}
        
    def _process_audio_input(self, audio_input: Any) -> Dict[str, Any]:
        """Conceptual placeholder for the Audio Encoder."""
        # This would be where conceptual_audio_encoder.py logic resides.
        return {"tone": "sarcastic", "multimodal_property": "pitch:high"}

    def encode_conceptual_vector(self, conceptual_summary: Dict[str, Any]) -> torch.Tensor:
        """
        Compresses the conceptual summary into a single, dense vector.
        """
        combined_embeddings = []
        for role, words in conceptual_summary.items():
            if role in self.concept_to_id:
                concept_id = self.concept_to_id[role]
                combined_embeddings.append(self.embedding_layer(torch.tensor(concept_id)))

            if isinstance(words, list):
                for word in words:
                    if word in self.prop_to_id:
                        prop_id = self.prop_to_id[word]
                        combined_embeddings.append(self.embedding_layer(torch.tensor(prop_id)))
            elif isinstance(words, str) and words in self.prop_to_id:
                prop_id = self.prop_to_id[words]
                combined_embeddings.append(self.embedding_layer(torch.tensor(prop_id)))

        if not combined_embeddings:
            return torch.zeros(self.embedding_dim)

        return torch.sum(torch.stack(combined_embeddings), dim=0)

if __name__ == '__main__':
    # Mocking a CKG instance for the demo
    class MockCKG:
        def query(self, entity_id: str):
            if entity_id == "cat":
                return {"node": {"type": "animal_agent"}}
            return None

    # Simple test of the encoder with multimodal input
    ckg_instance = MockCKG()
    encoder = ZenithConceptualEncoder(embedding_dim=128, ckg=ckg_instance)
    
    sentence = "The cat ran fast."
    visual_data = "a red ball"
    audio_data = "a high-pitched tone"
    
    encoded_vector = encoder(text_input=sentence, visual_input=visual_data, audio_input=audio_data)
    print(f"Fused vector shape: {encoded_vector.shape}")
