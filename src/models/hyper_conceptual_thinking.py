# /src/models/hyper_conceptual_thinking.py

import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from typing import Dict, List, Tuple
from ..conceptual_knowledge_graph.ckg import ConceptualKnowledgeGraph
import zenith_hct  # Import the compiled C++ module

class ConceptDiscoveryEngine:
    """
    The Concept Discovery Engine (CDE) is the core of the HCT layer.
    It analyzes the model's internal representations to identify emergent,
    high-value conceptual patterns and stores them persistently in the CKG.
    """
    def __init__(self, ckg: ConceptualKnowledgeGraph, num_clusters: int = 5, discovery_threshold: float = 0.5):
        self.ckg = ckg  # CKG instance
        self.num_clusters = num_clusters
        self.discovery_threshold = discovery_threshold
        self.kmeans_model = None
        self.concept_counter = 0
        # Initialize the C++-side MockCKG instance
        self.cpp_ckg = zenith_hct.MockCKG()
        self.relationship_types_counter = 0

    def analyze_for_new_concepts(self, fused_representation: torch.Tensor, confidence_score: float, reward: float, domain: str) -> Tuple[float, str | None]:
        """
        Analyzes the fused representation to see if it represents a newly discovered concept.
        
        Args:
            fused_representation: The output of the Conceptual Attention layer.
            confidence_score: The model's confidence in its current output.
            reward: The conceptual reward for the current state.
            domain: The current game domain.
        
        Returns:
            A tuple containing a discovery bonus and the name of the discovered concept (if any).
        """
        # New: Trigger HCT on a drop in confidence, indicating a novel or difficult problem.
        # This is a key upgrade to the HCT's purpose.
        if confidence_score < 0.3 or reward > self.discovery_threshold:
            print(f"HCT triggered. Confidence: {confidence_score}, Reward: {reward}")
            # The core logic for concept discovery and relationship creation is offloaded to the C++ backend.
            data_np = fused_representation.detach().cpu().numpy().reshape(1, -1)
            processed_data_np = np.asarray(zenith_hct.perform_hct_calculations(data_np, self.cpp_ckg))

            if self.kmeans_model is None:
                self.kmeans_model = MiniBatchKMeans(n_clusters=self.num_clusters, n_init=3).fit(processed_data_np)
                self.concept_counter += 1
                concept_name = f"HCT_Concept_{domain}_{self.concept_counter}"

                self.ckg.add_node(
                    concept_name,
                    {"type": "discovered_concept", "domain": domain, "reward": reward, "description": f"An emergent pattern in the {domain} domain."}
                )
                return 10.0, concept_name

            new_cluster = self.kmeans_model.predict(processed_data_np)[0]
            distances = self.kmeans_model.transform(processed_data_np)
            min_distance = np.min(distances)

            # New: Logic to propose and create new relationship types.
            # This is a fundamental change to the CKG's structure.
            if min_distance > 1.0:
                self.kmeans_model.partial_fit(processed_data_np)
                self.concept_counter += 1
                concept_name = f"HCT_Concept_{domain}_{self.concept_counter}"
                new_relationship_name = f"HCT_Relationship_{self.relationship_types_counter}"

                # New: Propose a new relationship type to the CKG.
                self.ckg.propose_new_relationship_type(new_relationship_name, "An emergent, novel relationship discovered by HCT.")
                self.relationship_types_counter += 1
                
                # Store the new concept and use the new relationship type.
                self.ckg.add_node(
                    concept_name,
                    {"type": "discovered_concept", "domain": domain, "reward": reward, "description": "A significantly different pattern from known concepts."}
                )
                self.ckg.add_edge(domain, concept_name, new_relationship_name)
                
                return 5.0, concept_name
        return 0.0, None

