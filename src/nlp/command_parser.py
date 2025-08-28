# /src/nlp/command_parser.py

import spacy
from typing import Dict, Any, List
from ..conceptual_knowledge_graph.ckg import ConceptualKnowledgeGraph

class CommandParser:
    """
    Parses natural language queries and extracts a structured command
    and its parameters using a conceptual understanding of the query.
    """
    def __init__(self, ckg: ConceptualKnowledgeGraph):
        self.ckg = ckg
        # Use a simplified conceptual mapping instead of a flat keyword list.
        self.command_map = {
            "explain": ["reasoning", "justification"],
            "strategy": ["plan", "goal"],
            "board_state": ["situation"],
            "eval_move": ["assess", "value"],
            "what_if": ["hypothetical", "simulate"]
        }
        # Pre-load the base ontology for a basic understanding
        self._initialize_conceptual_parser()

    def _initialize_conceptual_parser(self):
        """Initializes the parser by linking it to the CKG's base concepts."""
        for cmd, synonyms in self.command_map.items():
            for word in synonyms:
                self.ckg.add_node(word, {"type": "command_keyword", "command": cmd})
                self.ckg.add_edge(word, cmd, "IS_A_KEYWORD_FOR")

    def parse_command(self, query: str) -> Dict[str, Any]:
        """
        Takes a natural language query and returns a structured command
        by identifying and linking concepts from the CKG.
        
        Args:
            query (str): The user's input string.
            
        Returns:
            Dict: A dictionary containing the command and any relevant parameters.
        """
        lower_query = query.lower()
        
        # Step 1: Identify key concepts in the query using the CKG
        detected_concepts = []
        for word in lower_query.split():
            node_info = self.ckg.query(word)
            if node_info and node_info['node'].get('type') == 'command_keyword':
                detected_concepts.append(node_info['node'])

        # Step 2: Determine the command based on the detected concepts
        command = "unknown"
        for concept in detected_concepts:
            cmd = concept.get('command')
            if cmd:
                command = cmd
                break

        # Step 3: Extract entities based on the command context
        entities = {}
        tokens = lower_query.split()
        if 'move' in tokens:
            try:
                move_idx = tokens[tokens.index('move') + 1]
                entities['move'] = int(move_idx)
            except (ValueError, IndexError):
                pass

        return {
            "command": command,
            "entities": entities
        }
