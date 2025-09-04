# /src/web_access/web_access.py

import requests
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import time
import os
import re

# New: Import the optional C++ blockchain interface
try:
    import blockchain_interface_cpp
except ImportError:
    blockchain_interface_cpp = None
    print("Warning: Blockchain interface not found. WebAccess cannot verify records.")

class WebAccess:
    """
    The WebAccess module provides a structured interface for the Zenith Protocol
    to query real-time information from the internet. It handles search queries,
    retrieves results, and summarizes them for knowledge graph integration.
    """
    def __init__(self, ckg: Any, api_key: str = "DUMMY_API_KEY"):
        """Initializes the WebAccess module with a CKG instance.
        
        Args:
            ckg: An instance of the ConceptualKnowledgeGraph for data integration.
            api_key: A placeholder for a real search engine API key.
        """
        self.ckg = ckg
        self.api_key = api_key
        self.cache: Dict[str, Dict] = {} # A simple cache to avoid redundant searches
        self.blockchain_enabled = blockchain_interface_cpp is not None

    def _execute_search(self, query: str) -> Optional[Dict]:
        """
        Executes a web search query using a placeholder for a real search API.
        
        This is a dummy function to demonstrate the concept.
        In a production environment, this would call a real search API.
        """
        print(f"Executing web search for: '{query}'...")
        time.sleep(0.5)

        # Simple mock results for common queries
        mock_results = {
            "latest ai news": {
                "items": [
                    {"title": "Breakthrough in AI Causal Reasoning", "snippet": "Researchers at a major university have unveiled a new model focused on causal understanding, addressing the 'black box' problem."},
                    {"title": "Gemini 2.0 Launched by Google", "snippet": "Google has announced a major upgrade to its Gemini model, with enhanced performance and new features."},
                    {"title": "New record for LLM training", "snippet": "A new supercomputer has trained a massive language model in a fraction of the time, setting a new efficiency record."}
                ]
            },
            "what is a quantum computer": {
                "items": [
                    {"title": "Quantum Computing Explained", "snippet": "A quantum computer is a type of computer that uses the principles of quantum mechanics to perform computations."},
                    {"title": "The D-Wave Quantum System", "snippet": "D-Wave's quantum annealing system is a specialized machine for solving optimization problems."},
                    {"title": "Applications of Quantum Computing", "snippet": "Quantum computers can be used in drug discovery, materials science, and financial modeling."}
                ]
            }
        }

        return mock_results.get(query.lower(), {"items": []})
        
    def _scrape_data_from_source(self, url: str) -> str:
        """
        Simulates scraping content from a URL.
        In a real scenario, this would use a library like BeautifulSoup or Scrapy.
        """
        print(f"Scraping data from mock URL: {url}...")
        time.sleep(0.5)
        
        mock_html_content = (
            "<html><body><h1>This is a web page about AI.</h1>"
            "<p>A causal reasoning model helps an AI understand the 'why' behind events.</p>"
            "<p>This new approach is crucial for building trustworthy AI systems.</p></body></html>"
        )
        
        cleaned_text = re.sub('<[^>]*>', '', mock_html_content)
        return cleaned_text

    def search_and_summarize(self, query: str) -> Optional[str]:
        """
        Performs a web search, filters for relevance, and summarizes the results.
        It can also scrape content from a URL for a deeper dive.
        """
        # New: Check for a verifiable record on the blockchain first.
        if self.blockchain_enabled:
            print(f"[WebAccess] Checking for verifiable record for '{query}'...")
            verifiable_record = self.ckg.get_verifiable_record(query)
            if verifiable_record:
                print("[WebAccess] Found a verifiable record. Prioritizing over web search.")
                return verifiable_record['local_data']['node'].get('content', '')

        if query in self.cache:
            print("Retrieving from cache.")
            return self.cache[query]["summary"]

        search_results = self._execute_search(query)

        if not search_results or "items" not in search_results:
            print("No search results found.")
            return None
        
        relevant_content = []
        for item in search_results["items"]:
            if len(relevant_content) == 0 and query.lower() in item["title"].lower():
                full_text = self._scrape_data_from_source("[http://mock-url.com](http://mock-url.com)")
                relevant_content.append(full_text)
            else:
                relevant_content.append(item["snippet"])

        if not relevant_content:
            return None

        summary = " ".join(relevant_content)

        self.cache[query] = {
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
        
        print("Search results summarized and cached.")
        return summary

    def check_for_update(self, query: str, time_limit_minutes: int = 60) -> bool:
        """
        Checks if a cached result is outdated and needs a fresh search.
        """
        if query not in self.cache:
            return True

        cached_time = datetime.fromisoformat(self.cache[query]["timestamp"])
        time_since_cached = (datetime.now() - cached_time).total_seconds() / 60

        return time_since_cached > time_limit_minutes

# --- Example Usage ---
if __name__ == '__main__':
    from ..conceptual_knowledge_graph.ckg import ConceptualKnowledgeGraph
    mock_ckg = ConceptualKnowledgeGraph()
    web_access = WebAccess(ckg=mock_ckg)

    print("--- First Search ---")
    query_1 = "latest AI news"
    info = web_access.search_and_summarize(query_1)
    if info:
        print(f"\nZenith received this information:\n'{info}'")
    else:
        print("\nNo relevant information received.")

    print("\n--- Second Search (from cache) ---")
    info_cached = web_access.search_and_summarize(query_1)
    if info_cached:
        print(f"\nZenith received this information:\n'{info_cached}'")

    print("\n--- Checking for update (simulating time passing) ---")
    web_access.cache[query_1]["timestamp"] = (datetime.now() - timedelta(minutes=90)).isoformat()

    needs_update = web_access.check_for_update(query_1)
    print(f"Does the knowledge for '{query_1}' need an update? {needs_update}")