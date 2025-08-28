# /src/conceptual_knowledge_graph/ckg.py

import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

class ConceptualKnowledgeGraph:
    """
    An in-memory, graph-based knowledge store for the Zenith Protocol.
    It stores concepts, properties, prompts, and responses as nodes, and
    their relationships as directed edges. This serves as the model's
    long-term memory.
    """
    def __init__(self, storage_path: str = "conceptual_graph.json"):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: Dict[str, Dict[str, Any]] = {}
        self.storage_path = storage_path
        self._load_graph()

    def _load_graph(self):
        """Loads the graph from a JSON file if it exists."""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.nodes = data.get("nodes", {})
                self.edges = data.get("edges", {})
            print(f"Conceptual Knowledge Graph loaded from {self.storage_path}")
        except (FileNotFoundError, json.JSONDecodeError):
            print("No existing graph found. Starting with a new one.")
            self._initialize_base_concepts()
            self._save_graph()

    def _save_graph(self):
        """Saves the current state of the graph to a JSON file."""
        with open(self.storage_path, 'w') as f:
            json.dump({"nodes": self.nodes, "edges": self.edges}, f, indent=4)
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
            self.add_node(node_id, data)

    def add_node(self, node_id: str, properties: Dict[str, Any]) -> None:
        """Adds or updates a node in the graph."""
        if node_id in self.nodes and self.nodes[node_id].get("type") != properties.get("type"):
            print(f"Warning: Node '{node_id}' already exists with a different type. Skipping.")
            return

        self.nodes[node_id] = {
            **self.nodes.get(node_id, {}),
            **properties,
            "last_updated": datetime.now().isoformat()
        }
    
    def add_edge(self, source_id: str, target_id: str, relationship: str, properties: Optional[Dict[str, Any]] = None) -> None:
        """Adds a directed edge between two nodes."""
        if source_id not in self.nodes or target_id not in self.nodes:
            print(f"Error: One or both nodes ('{source_id}', '{target_id}') do not exist.")
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
        """Updates properties of an existing node."""
        if node_id in self.nodes:
            self.nodes[node_id].update(new_properties)
            self.nodes[node_id]["last_updated"] = datetime.now().isoformat()
        else:
            print(f"Error: Node '{node_id}' not found.")

    def query(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Returns the node and its immediate connections."""
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
    
    def add_prompt_response(self, prompt: str, response: str, concepts: List[str]) -> None:
        """Adds a prompt and response to the graph and links them to concepts."""
        prompt_id = f"Prompt_{len(self.nodes) + 1}"
        response_id = f"Response_{len(self.nodes) + 2}"
        
        self.add_node(prompt_id, {"type": "prompt", "content": prompt})
        self.add_node(response_id, {"type": "response", "content": response})
        
        self.add_edge(prompt_id, response_id, "GENERATED_RESPONSE", {"timestamp": datetime.now().isoformat()})
        
        for concept in concepts:
            if concept in self.nodes:
                self.add_edge(prompt_id, concept, "REFERENCES")
                self.add_edge(response_id, concept, "CONTAINS")
            else:
                self.add_node(concept, {"type": "concept", "source": "conversation"})
                self.add_edge(prompt_id, concept, "DISCOVERED")
                self.add_edge(response_id, concept, "CONTAINS")
        
        self._save_graph()

# --- Example Usage ---
if __name__ == '__main__':
    ckg = ConceptualKnowledgeGraph()
    
    # Add a new concept with properties
    ckg.add_node("AI", {"type": "concept", "properties": {"nature": "computational", "purpose": "problem solving"}})
    ckg.add_edge("AI", "Agent", "IS_A")
    
    # Add a conversation and learn from it
    prompt_text = "What is a good way to stay fit?"
    response_text = "Running is a great option because it helps burn calories."
    concepts_in_conversation = ["run", "calories", "fit"]
    
    ckg.add_prompt_response(prompt_text, response_text, concepts_in_conversation)

    # Query the graph to see what it knows about "run"
    run_info = ckg.query("run")
    print("\n--- Querying 'run' ---")
    print(json.dumps(run_info, indent=4))
    
    # Query the graph to see what it knows about the latest prompt
    prompt_info = ckg.query("Prompt_1")
    print("\n--- Querying 'Prompt_1' ---")
    print(json.dumps(prompt_info, indent=4))
    
    # Verify that the graph was saved
    ckg_reloaded = ConceptualKnowledgeGraph()
    print("\nGraph reloaded from file to verify persistence.")
    reloaded_info = ckg_reloaded.query("run")
    print(json.dumps(reloaded_info, indent=4))

