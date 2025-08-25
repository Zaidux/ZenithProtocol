# /src/models/hyper_conceptual_thinking.py

import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from typing import Dict, List, Tuple

class ConceptDiscoveryEngine:
    """
    The Concept Discovery Engine (CDE) is the core of the HCT layer.
    It analyzes the model's internal representations to identify emergent,
    high-value conceptual patterns not present in the original input.
    """
    def __init__(self, num_clusters: int = 5, discovery_threshold: float = 0.5):
        self.num_clusters = num_clusters
        self.discovery_threshold = discovery_threshold
        self.kmeans_model = None
        self.concept_counter = 0
        self.concept_names = [] # To store names of discovered concepts

    def analyze_for_new_concepts(self, fused_representation: torch.Tensor, reward: float, domain: str) -> Tuple[float, str | None]:
        """
        Analyzes the fused representation to see if it represents a newly discovered concept.
        
        Args:
            fused_representation: The output of the Conceptual Attention layer.
            reward: The conceptual reward for the current state.
            domain: The current game domain.
        
        Returns:
            A tuple containing a discovery bonus and the name of the discovered concept (if any).
        """
        # We only look for new concepts in high-reward states
        if reward < self.discovery_threshold:
            return 0.0, None

        # Prepare the data for clustering
        data = fused_representation.detach().cpu().numpy().reshape(1, -1)
        
        if self.kmeans_model is None:
            # First time, initialize the model with a single point
            self.kmeans_model = MiniBatchKMeans(n_clusters=self.num_clusters, n_init=3).fit(data)
            self.concept_counter += 1
            concept_name = f"HCT_Concept_{domain}_{self.concept_counter}"
            self.concept_names.append(concept_name)
            return 10.0, concept_name # A high bonus for the first discovery
        
        # Predict which cluster the new representation belongs to
        new_cluster = self.kmeans_model.predict(data)[0]
        
        # Check if this state is significantly different from existing clusters
        distances = self.kmeans_model.transform(data)
        min_distance = np.min(distances)
        
        # If the minimum distance is high, it could be a new concept
        if min_distance > 1.0: # A heuristic threshold
            # Retrain the model to include the new concept
            self.kmeans_model.partial_fit(data)
            self.concept_counter += 1
            concept_name = f"HCT_Concept_{domain}_{self.concept_counter}"
            self.concept_names.append(concept_name)
            return 5.0, concept_name
            
        return 0.0, None
