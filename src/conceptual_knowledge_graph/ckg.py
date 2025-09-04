# /src/conceptual_knowledge_graph/ckg.py

import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import hashlib

# Assuming a mock ConceptualEncoder and an in-memory graph database
# In a real-world scenario, these would be separate, optimized modules.
# We'll simulate them here for demonstration.
from ..conceptual_encoder.conceptual_encoder import ConceptualEncoder
from .in_memory_db import InMemoryGraphDB

# New: Import the optional C++ blockchain interface
try:
    import blockchain_interface_cpp
except ImportError:
    blockchain_interface_cpp = None
    print("Warning: Blockchain interface not found. Running in local-only mode.")

class ConceptualKnowledgeGraph:
    """
    An in-memory, graph-based knowledge store for the Zenith Protocol.
    It stores concepts, properties, prompts, and responses as nodes, and
    their relationships as directed edges. This serves as the model's
    long-term memory.
    """
    def __init__(self, storage_path: str = "conceptual_graph.json"):
        # We will use a dedicated in-memory graph database instead of a simple dictionary.
        # This simulates a more performant, scalable solution.
        self.db = InMemoryGraphDB()
        self.storage_path = storage_path
        self._load_graph()

        # New: Initialize the optional blockchain interface
        self.blockchain_enabled = blockchain_interface_cpp is not None
        if self.blockchain_enabled:
            self.blockchain_client = blockchain_interface_cpp.BlockchainInterface()

    def _load_graph(self):
        """Loads the graph from a JSON file if it exists."""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.db.nodes = data.get("nodes", {})
                self.db.edges = data.get("edges", {})
            print(f"Conceptual Knowledge Graph loaded from {self.storage_path}")
        except (FileNotFoundError, json.JSONDecodeError):
            print("No existing graph found. Starting with a new one.")
            self._initialize_base_concepts()
            self._save_graph()

    def _save_graph(self):
        """Saves the current state of the graph to a JSON file."""
        # New: Store a verifiable record on the blockchain if enabled
        if self.blockchain_enabled:
            data_to_store = json.dumps({"nodes": self.db.nodes, "edges": self.db.edges}, sort_keys=True)
            self.blockchain_client.add_to_blockchain(data_to_store)
            print("Graph state hashed and stored on the mock blockchain.")

        with open(self.storage_path, 'w') as f:
            json.dump({"nodes": self.db.nodes, "edges": self.db.edges}, f, indent=4)
        print(f"Conceptual Knowledge Graph saved to {self.storage_path}")

    def _initialize_base_concepts(self):
        """Initializes a few core concepts to get the graph started."""
        base_concepts = {
            "Agent": {"type": "concept", "description": "Entity that performs actions."},
            "Action": {"type": "concept", "description": "A verb or process executed."},
            "Object": {"type": "concept", "description": "A tangible item with properties."},
            "Reason": {"type": "concept", "description": "The 'why' behind an action."},
            "Internet": {"type": "concept", "description": "A global network of computers."},
            "Real-Time Data": {"type": "concept", "description": "Information updated constantly."},
        }
        for node_id, data in base_concepts.items():
            # Add initial nodes with a default high confidence score
            self.add_node(node_id, {**data, "verifiability_score": 1.0})

    def add_node(self, node_id: str, properties: Dict[str, Any]) -> None:
        """Adds or updates a node in the graph."""
        self.db.add_node(node_id, properties)
        # New: Automatically save after adding a node for a verifiable ledger
        self._save_graph()

    def add_edge(self, source_id: str, target_id: str, relationship: str, properties: Optional[Dict[str, Any]] = None) -> None:
        """Adds a directed edge between two nodes."""
        self.db.add_edge(source_id, target_id, relationship, properties)
        # New: Automatically save after adding an edge
        self._save_graph()

    def update_node_properties(self, node_id: str, new_properties: Dict[str, Any]) -> None:
        """Updates properties of an existing node."""
        self.db.update_node_properties(node_id, new_properties)
        self._save_graph()

    def query(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Returns the node and its immediate connections."""
        return self.db.query(entity_id)

    def add_prompt_response(self, prompt: str, response: str, conceptual_encoder) -> None:
        """
        Processes a prompt and response using the Conceptual Encoder
        and adds the rich conceptual data to the graph.
        """
        # Step 1: Use the Conceptual Encoder to extract a rich set of concepts and relations.
        extracted_data = conceptual_encoder.extract_concepts_and_relations(prompt, response)

        # Step 2: Add nodes and edges to the graph based on the extracted data.
        # This includes Agents, Actions, Objects, Reasons, and their relationships.
        for data in extracted_data:
            source_id = data.get("source_id")
            source_type = data.get("source_type")
            target_id = data.get("target_id")
            target_type = data.get("target_type")
            relationship = data.get("relationship")
            properties = data.get("properties", {})

            # Initialize a confidence score for newly discovered concepts.
            if not self.db.node_exists(source_id):
                self.add_node(source_id, {"type": source_type, "content": source_id, "verifiability_score": 0.5})

            if not self.db.node_exists(target_id):
                self.add_node(target_id, {"type": target_type, "content": target_id, "verifiability_score": 0.5})

            self.add_edge(source_id, target_id, relationship, properties)

            # Update the verifiability score of referenced concepts.
            # This simulates the "dynamic updating based on how often it's referenced" idea.
            self.db.update_verifiability_score(source_id, 0.1)
            self.db.update_verifiability_score(target_id, 0.1)

        self._save_graph()

    def get_verifiability_score(self, node_id: str) -> float:
        """Returns the verifiability score of a given node."""
        node = self.db.query(node_id)
        if node and "node" in node and "verifiability_score" in node["node"]:
            return node["node"]["verifiability_score"]
        return 0.0

    # New: Add a method to retrieve a verifiable record from the blockchain
    def get_verifiable_record(self, data_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a verifiable record from the blockchain and returns it.
        """
        if self.blockchain_enabled:
            # Hash the data_id to find the record on the blockchain
            record_hash = self.blockchain_client.generate_hash(data_id)
            tx = self.blockchain_client.get_from_blockchain(record_hash)
            
            if tx.data_hash:
                # The data itself is not stored on the blockchain, only the hash.
                # We can verify that the local record matches the hash on the blockchain.
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
    """A mock Conceptual Encoder to simulate semantic compression."""
    def extract_concepts_and_relations(self, prompt: str, response: str) -> List[Dict[str, Any]]:
        return [
            {"source_id": "run", "source_type": "concept", "target_id": "Action", "target_type": "concept", "relationship": "IS_A", "properties": {"motion_type": "continuous"}},
        ]

class InMemoryGraphDB:
    """A mock in-memory graph database with basic functionality."""
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

# --- Example Usage ---
if __name__ == '__main__':
    # Initialize the graph
    ckg = ConceptualKnowledgeGraph()
    conceptual_encoder = ConceptualEncoder()

    # Add a conversation with the new, enhanced method
    prompt_text = "What is a good way to stay fit?"
    response_text = "Running is a great option because it helps burn calories."
    ckg.add_prompt_response(prompt_text, response_text, conceptual_encoder)

    # Query the graph to see what it knows about "run"
    run_info = ckg.query("run")
    print("\n--- Querying 'run' ---")
    print(json.dumps(run_info, indent=4))

    # Get the verifiability score
    run_score = ckg.get_verifiability_score("run")
    print(f"\nThe verifiability score for 'run' is: {run_score}")

    # Verify that the graph was saved and reloaded
    ckg_reloaded = ConceptualKnowledgeGraph()
    print("\nGraph reloaded from file to verify persistence.")
    reloaded_info = ckg_reloaded.query("run")
    print(json.dumps(reloaded_info, indent=4))