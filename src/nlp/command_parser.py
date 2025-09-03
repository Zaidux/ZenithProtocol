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
            "explain": ["reasoning", "justification", "explain"],
            "strategy": ["plan", "goal", "strategy"],
            "board_state": ["situation"],
            "eval_move": ["assess", "value", "evaluate"],
            "what_if": ["hypothetical", "simulate"]
        }
        # A simple NLP model is loaded to handle part-of-speech tagging.
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("Spacy model not found. Please run: python -m spacy download en_core_web_sm")
            self.nlp = None

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
        if not self.nlp:
            return {"command": "error", "entities": {}, "error_message": "Spacy model not loaded."}
        
        doc = self.nlp(query.lower())
        command = "unknown"
        entities = {}

        # Step 1: Identify the command based on conceptual mapping
        for token in doc:
            if token.text in self.command_map:
                command = token.text
                break
            # Check for synonyms in the CKG
            node_info = self.ckg.query(token.text)
            if node_info and node_info['node'].get('type') == 'command_keyword':
                command = node_info['node'].get('command')
                break

        # Step 2: Extract entities with improved logic
        if command in ["eval_move", "what_if"]:
            # This logic can be enhanced with more sophisticated entity recognition.
            for token in doc:
                # Assuming numbers are moves
                if token.like_num:
                    entities['move'] = int(token.text)
                # Check for chess move notation (e.g., 'e4', 'Nf3')
                elif command == "eval_move" and len(token.text) == 2 and token.text[0].isalpha() and token.text[1].isdigit():
                    entities['move'] = token.text
        
        # This is for commands with a specific structure, like "move pawn to c3"
        elif command == "move":
            for token in doc:
                if token.dep_ == 'dobj' or token.dep_ == 'pobj':
                    entities['object'] = token.text
                if token.text == 'to' and token.head.text == 'move':
                    entities['destination'] = token.text
        
        # Step 3: Use the CKG for deeper entity resolution
        if entities and 'move' in entities:
            move_entity = entities['move']
            move_node = self.ckg.query(str(move_entity))
            if move_node and move_node['node'].get('type') == 'move':
                # This could be used for advanced logic, like checking if the move is legal.
                print(f"Recognized move '{move_entity}' as a known concept in the CKG.")

        return {
            "command": command,
            "entities": entities
        }
