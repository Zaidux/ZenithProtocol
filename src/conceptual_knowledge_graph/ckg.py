# /src/conceptual_knowledge_graph/ckg.py

import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import hashlib

from ..conceptual_encoder.conceptual_encoder import ZenithConceptualEncoder
from .in_memory_db import InMemoryGraphDB

try:
    import blockchain_interface_cpp
except ImportError:
    blockchain_interface_cpp = None
    print("Warning: Blockchain interface not found. Running in local-only mode.")

class ConceptualKnowledgeGraph:
    """
    An in-memory, graph-based knowledge store for the Zenith Protocol.
    It now handles multimodal, socio-linguistic, and architectural properties.
    """
    def __init__(self, storage_path: str = "conceptual_graph.json"):
        self.db = InMemoryGraphDB()
        self.storage_path = storage_path
        self._load_graph()
        # New: Add a counter for proposals to ensure unique IDs
        self.proposal_counter = 0

        self.blockchain_enabled = blockchain_interface_cpp is not None
        if self.blockchain_enabled:
            self.blockchain_client = blockchain_interface_cpp.BlockchainInterface()

        self.relationship_types = self._get_default_relationship_types()

    def _get_default_relationship_types(self) -> Dict[str, Dict[str, Any]]:
        return {
            "IS_A": {"description": "A conceptual inheritance."},
            "HAS_PROPERTY": {"description": "Links a concept to its property."},
            "PERFORMS": {"description": "Links an Agent to an Action."},
            "ACTS_ON": {"description": "Links an Action to an Object."},
            "HAS_REASON": {"description": "Links an Action to its Reason."},
            "HAS_DISCOVERED_CONCEPT": {"description": "Links a domain to a discovered concept."},
            "IS_VISUAL": {"description": "Links a concept to a visual element."},
            "IS_AUDIO": {"description": "Links a concept to an audio element."},
            "HAS_TONE": {"description": "Links a concept to a socio-linguistic tone."},
            "PROPOSED_UPGRADE": {"description": "Links a component to an architectural upgrade proposal."}, # New
            "FINALIZED": {"description": "Links a human decision to an architectural proposal."}, # New
        }

    def _load_graph(self):
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.db.nodes = data.get("nodes", {})
                self.db.edges = data.get("edges", {})
                self.relationship_types = data.get("relationship_types", self._get_default_relationship_types())
            print(f"Conceptual Knowledge Graph loaded from {self.storage_path}")
        except (FileNotFoundError, json.JSONDecodeError):
            print("No existing graph found. Starting with a new one.")
            self._initialize_base_concepts()
            self._save_graph()

    def _save_graph(self):
        if self.blockchain_enabled:
            data_to_store = json.dumps({
                "nodes": self.db.nodes,
                "edges": self.db.edges,
                "relationship_types": self.relationship_types
            }, sort_keys=True)
            self.blockchain_client.add_to_blockchain(data_to_store)
            print("Graph state hashed and stored on the mock blockchain.")

        with open(self.storage_path, 'w') as f:
            json.dump({
                "nodes": self.db.nodes,
                "edges": self.db.edges,
                "relationship_types": self.relationship_types
            }, f, indent=4)
        print(f"Conceptual Knowledge Graph saved to {self.storage_path}")

    def _initialize_base_concepts(self):
        base_concepts = {
            "Agent": {"type": "concept", "description": "Entity that performs actions."},
            "Action": {"type": "concept", "description": "A verb or process executed."},
            "Object": {"type": "concept", "description": "A tangible item with properties."},
            "Reason": {"type": "concept", "description": "The 'why' behind an action."},
            "Internet": {"type": "concept", "description": "A global network of computers."},
            "Real-Time Data": {"type": "concept", "description": "Information updated constantly."},
            "Visual_Data": {"type": "modality", "description": "Represents information from an image or video."},
            "Audio_Data": {"type": "modality", "description": "Represents information from an audio clip."},
            "Socio-Linguistic_Context": {"type": "modality", "description": "Represents conversational tone and style."},
        }
        for node_id, data in base_concepts.items():
            self.add_node(node_id, {**data, "verifiability_score": 1.0})

    def add_node(self, node_id: str, properties: Dict[str, Any]) -> None:
        self.db.add_node(node_id, properties)
        if node_id.startswith("proposal_"):
            self.proposal_counter += 1
        self._save_graph()

    def add_edge(self, source_id: str, target_id: str, relationship: str, properties: Optional[Dict[str, Any]] = None) -> None:
        if relationship not in self.relationship_types:
            print(f"Warning: Relationship type '{relationship}' is not in the schema. Adding a default entry.")
            self.propose_new_relationship_type(relationship, "Dynamically created relationship.")

        self.db.add_edge(source_id, target_id, relationship, properties)
        self._save_graph()

    def update_node_properties(self, node_id: str, new_properties: Dict[str, Any]) -> None:
        self.db.update_node_properties(node_id, new_properties)
        self._save_graph()

    def query(self, entity_id: str) -> Optional[Dict[str, Any]]:
        return self.db.query(entity_id)

    def query_by_property(self, key: str, value: Any) -> List[Dict[str, Any]]:
        """
        New Method: Queries the graph for all nodes that match a specific key-value pair in their properties.
        This is crucial for finding all pending architectural proposals.
        """
        matching_nodes = []
        for node_id, node_data in self.db.nodes.items():
            if node_data.get(key) == value:
                # Return the full node data with its connections.
                matching_nodes.append(self.query(node_id))
        return matching_nodes

    def add_prompt_response(self, prompt: str, response: str, conceptual_encoder: 'ZenithConceptualEncoder') -> None:
        extracted_data = conceptual_encoder.extract_concepts_and_relations(prompt, response)

        for data in extracted_data:
            source_id = data.get("source_id")
            source_type = data.get("source_type")
            target_id = data.get("target_id")
            target_type = data.get("target_type")
            relationship = data.get("relationship")
            properties = data.get("properties", {})

            if not self.db.node_exists(source_id):
                self.add_node(source_id, {"type": source_type, "content": source_id, "verifiability_score": 0.5})
            if not self.db.node_exists(target_id):
                self.add_node(target_id, {"type": target_type, "content": target_id, "verifiability_score": 0.5})

            self.add_edge(source_id, target_id, relationship, properties)
            self.db.update_verifiability_score(source_id, 0.1)
            self.db.update_verifiability_score(target_id, 0.1)

        self._save_graph()

    def propose_new_relationship_type(self, relationship_name: str, description: str, is_vulnerability: bool = False):
        if relationship_name not in self.relationship_types:
            self.relationship_types[relationship_name] = {
                "description": description,
                "is_vulnerability": is_vulnerability,
                "created_by": "HCT" if not is_vulnerability else "AdversarialModule",
                "timestamp": datetime.now().isoformat()
            }
            print(f"Proposed new relationship type: '{relationship_name}'")
            self._save_graph()

    def get_verifiability_score(self, node_id: str) -> float:
        node = self.db.query(node_id)
        if node and "node" in node and "verifiability_score" in node["node"]:
            return node["node"]["verifiability_score"]
        return 0.0

    def get_verifiable_record(self, data_id: str) -> Optional[Dict[str, Any]]:
        if self.blockchain_enabled:
            record_hash = self.blockchain_client.generate_hash(data_id)
            tx = self.blockchain_client.get_from_blockchain(record_hash)

            if tx.data_hash:
                local_record = self.query(data_id)
                local_data_string = json.dumps(local_record, sort_keys=True)
                if self.blockchain_client.verify_transaction(tx, local_data_string):
                    print(f"Verifiable record for '{data_id}' found on blockchain.")
                    return {"local_data": local_record, "blockchain_record": tx}
                else:
                    print(f"Warning: Local data for '{data_id}' does not match blockchain record.")
                    return None
        return None

# --- Mock Classes for Demonstration ---
class ConceptualEncoder:
    def extract_concepts_and_relations(self, prompt: str, response: str) -> List[Dict[str, Any]]:
        return [
            {"source_id": "run", "source_type": "concept", "target_id": "Action", "target_type": "concept", "relationship": "IS_A", "properties": {"motion_type": "continuous"}},
        ]

class InMemoryGraphDB:
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def add_node(self, node_id: str, properties: Dict[str, Any]) -> None:
        if node_id in self.nodes and self.nodes[node_id].get("type") != properties.get("type"):
            return
        self.nodes[node_id] = {
            **self.nodes.get(node_id, {}),
            **properties,
            "last_updated": datetime.now().isoformat()
        }

    def add_edge(self, source_id: str, target_id: str, relationship: str, properties: Optional[Dict[str, Any]] = None) -> None:
        if source_id not in self.nodes or target_id not in self.nodes:
            return
        edge_id = f"{source_id}_{relationship}_{target_id}"
        self.edges[edge_id] = {
            "source": source_id,
            "target": target_id,
            "relationship": relationship,
            "properties": properties or {},
            "created_at": datetime.now().isoformat()
        }

    def update_node_properties(self, node_id: str, new_properties: Dict[str, Any]) -> None:
        if node_id in self.nodes:
            self.nodes[node_id].update(new_properties)
            self.nodes[node_id]["last_updated"] = datetime.now().isoformat()
        else:
            pass

    def query(self, entity_id: str) -> Optional[Dict[str, Any]]:
        if entity_id not in self.nodes:
            return None
        node_data = self.nodes.get(entity_id, {})
        related_nodes = []
        for edge_id, edge in self.edges.items():
            if edge["source"] == entity_id:
                related_nodes.append({
                    "relationship": edge["relationship"],
                    "target_id": edge["target"],
                    "target_data": self.nodes.get(edge["target"]),
                    "edge_properties": edge["properties"]
                })
            elif edge["target"] == entity_id:
                related_nodes.append({
                    "relationship": edge["relationship"],
                    "source_id": edge["source"],
                    "source_data": self.nodes.get(edge["source"]),
                    "edge_properties": edge["properties"]
                })
        return {"node": node_data, "connections": related_nodes}

    def node_exists(self, node_id: str) -> bool:
        return node_id in self.nodes

    def update_verifiability_score(self, node_id: str, score_change: float) -> None:
        if node_id in self.nodes and "verifiability_score" in self.nodes[node_id]:
            current_score = self.nodes[node_id]["verifiability_score"]
            new_score = max(0.0, min(1.0, current_score + score_change))
            self.nodes[node_id]["verifiability_score"] = new_score
            self.nodes[node_id]["last_updated"] = datetime.now().isoformat()

# Example usage
if __name__ == '__main__':
    ckg = ConceptualKnowledgeGraph()
    conceptual_encoder = ConceptualEncoder()

    # Add a conversation
    prompt_text = "What is a good way to stay fit?"
    response_text = "Running is a great option because it helps burn calories."
    ckg.add_prompt_response(prompt_text, response_text, conceptual_encoder)

    # Add a multimodal concept
    ckg.add_node("red_car", {"type": "visual_concept", "modality": "visual", "properties": {"color": "red"}})
    ckg.add_node("high_pitch", {"type": "audio_concept", "modality": "audio", "properties": {"pitch": "high"}})
    ckg.add_edge("red_car", "Visual_Data", "IS_VISUAL")
    ckg.add_edge("high_pitch", "Audio_Data", "IS_AUDIO")
    ckg.add_edge("Running", "Socio-Linguistic_Context", "HAS_TONE", {"value": "positive"})

    # Query the graph
    run_info = ckg.query("run")
    print("\n--- Querying 'run' ---")
    print(json.dumps(run_info, indent=4))
