# src/nlp/command_parser.py

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
            "what_if": ["hypothetical", "simulate"],
            "organize_files": ["gather", "organize", "collect"] # New: Command for local tasks
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
            node_info = self.ckg.query(token.text)
            if node_info and node_info['node'].get('type') == 'command_keyword':
                command = node_info['node'].get('command')
                break

        # Step 2: Extract entities with improved logic
        if command in ["eval_move", "what_if"]:
            for token in doc:
                if token.like_num:
                    entities['move'] = int(token.text)
                elif command == "eval_move" and len(token.text) == 2 and token.text[0].isalpha() and token.text[1].isdigit():
                    entities['move'] = token.text
        
        # New: Logic for extracting entities for on-device tasks
        if command == "organize_files":
            folder_name = None
            file_types = []
            conceptual_filters = []
            
            # Look for a folder name (a noun with "folder" or "named")
            for i, token in enumerate(doc):
                if token.text == 'folder' and i + 1 < len(doc) and doc[i+1].text == 'named':
                    folder_name = doc[i+2].text.strip('"')
                elif 'photos' in token.text or 'images' in token.text:
                    file_types.extend(['.jpg', '.png', '.jpeg'])
                elif 'dog' in token.text or 'cat' in token.text:
                    conceptual_filters.append('pets')
            
            entities = {
                "folder_name": folder_name,
                "file_types": file_types,
                "conceptual_filters": conceptual_filters
            }
            
        return {
            "command": command,
            "entities": entities
        }
