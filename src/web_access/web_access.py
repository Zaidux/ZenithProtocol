# /src/web_access/web_access.py

"""
Enhanced Web Access with Verifiable Knowledge Integration
=========================================================
Adds advanced caching, credibility scoring, and blockchain verification.
"""

import requests
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import time
import os
import re
from urllib.parse import urlparse
import hashlib

# New: Import the optional C++ blockchain interface
try:
    import blockchain_interface_cpp
except ImportError:
    blockchain_interface_cpp = None
    print("Warning: Blockchain interface not found. WebAccess cannot verify records.")

class WebAccess:
    """
    Enhanced WebAccess module with credibility scoring, advanced caching,
    and blockchain-based verification for trustworthy knowledge integration.
    """
    def __init__(self, ckg: Any, api_key: str = "DUMMY_API_KEY"):
        """Initializes the WebAccess module with enhanced capabilities."""
        self.ckg = ckg
        self.api_key = api_key
        self.cache: Dict[str, Dict] = {}
        self.source_credibility = self._initialize_credibility_db()
        self.blockchain_enabled = blockchain_interface_cpp is not None
        self.query_history = []

    def _initialize_credibility_db(self) -> Dict[str, float]:
        """Initialize source credibility database."""
        return {
            "arxiv.org": 0.95,
            "nature.com": 0.93,
            "sciencedirect.com": 0.90,
            "wikipedia.org": 0.85,
            "medium.com": 0.70,
            "personal.blog": 0.60,
            "unknown": 0.50
        }

    def _get_source_credibility(self, url: str) -> float:
        """Get credibility score for a source."""
        domain = urlparse(url).netloc.lower()
        return self.source_credibility.get(domain, self.source_credibility["unknown"])

    def _execute_advanced_search(self, query: str) -> Dict:
        """
        Enhanced search with credibility scoring and diversity.
        """
        print(f"Executing advanced web search for: '{query}'...")
        time.sleep(0.5)

        # Mock results with credibility metadata
        mock_results = {
            "latest ai news": {
                "items": [
                    {
                        "title": "Breakthrough in AI Causal Reasoning",
                        "snippet": "Researchers at Stanford unveiled a new causal reasoning model...",
                        "url": "https://arxiv.org/abs/2401.12345",
                        "source": "arxiv.org",
                        "date": "2024-01-15"
                    },
                    {
                        "title": "Google's Gemini 2.0 Launch",
                        "snippet": "Google announced major upgrades to Gemini with new capabilities...",
                        "url": "https://techcrunch.com/2024/01/gemini-2",
                        "source": "techcrunch.com", 
                        "date": "2024-01-14"
                    }
                ]
            }
        }

        results = mock_results.get(query.lower(), {"items": []})
        
        # Add credibility scores
        for item in results["items"]:
            item["credibility"] = self._get_source_credibility(item["url"])
        
        return results

    def _generate_verifiable_hash(self, content: str) -> str:
        """Generate hash for content verification."""
        return hashlib.sha256(content.encode()).hexdigest()

    def search_and_summarize(self, query: str, min_credibility: float = 0.7) -> Optional[Dict]:
        """
        Enhanced search with credibility filtering and verification.
        """
        # Track query history
        self.query_history.append({
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "min_credibility": min_credibility
        })

        # Check blockchain first for verified knowledge
        if self.blockchain_enabled:
            blockchain_result = self._check_blockchain_knowledge(query)
            if blockchain_result and blockchain_result["credibility"] >= min_credibility:
                return blockchain_result

        # Check cache with credibility consideration
        cache_key = f"{query}_{min_credibility}"
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if not self._is_cache_expired(cached["timestamp"]):
                return cached["result"]

        # Execute search
        search_results = self._execute_advanced_search(query)
        
        # Filter by credibility
        credible_items = [
            item for item in search_results.get("items", [])
            if item.get("credibility", 0) >= min_credibility
        ]

        if not credible_items:
            print(f"No results meeting credibility threshold {min_credibility}")
            return None

        # Generate verifiable summary
        summary = self._generate_verifiable_summary(credible_items, query)
        
        # Store in cache
        self.cache[cache_key] = {
            "result": summary,
            "timestamp": datetime.now().isoformat(),
            "credibility_score": min([item["credibility"] for item in credible_items])
        }

        # Add to CKG with verification data
        self._integrate_to_ckg(query, summary, credible_items)

        return summary

    def _generate_verifiable_summary(self, items: List[Dict], query: str) -> Dict:
        """Generate a verifiable summary with source attribution."""
        summary_text = f"Information about '{query}':\n\n"
        
        for i, item in enumerate(items, 1):
            summary_text += f"{i}. {item['snippet']} [Source: {item['source']}, Credibility: {item['credibility']:.2f}]\n"
        
        verification_hash = self._generate_verifiable_hash(summary_text)
        
        return {
            "content": summary_text,
            "sources": [item["url"] for item in items],
            "average_credibility": sum(item["credibility"] for item in items) / len(items),
            "verification_hash": verification_hash,
            "generated_at": datetime.now().isoformat()
        }

    def _integrate_to_ckg(self, query: str, summary: Dict, sources: List[Dict]):
        """Integrate search results into CKG with verification."""
        # Create knowledge node
        knowledge_id = f"web_knowledge_{hashlib.sha256(query.encode()).hexdigest()[:8]}"
        
        self.ckg.add_node(knowledge_id, {
            "type": "web_knowledge",
            "content": summary["content"],
            "query": query,
            "sources": sources,
            "credibility": summary["average_credibility"],
            "verification_hash": summary["verification_hash"],
            "timestamp": summary["generated_at"]
        })
        
        # Link to related concepts
        query_concepts = query.split()
        for concept in query_concepts:
            if len(concept) > 3:  # Avoid short words
                self.ckg.add_edge(concept, knowledge_id, "RELATED_TO")

    def _check_blockchain_knowledge(self, query: str) -> Optional[Dict]:
        """Check blockchain for verified knowledge."""
        try:
            # This would interface with actual blockchain in production
            record = self.ckg.get_verifiable_record(query)
            if record and record["local_data"]:
                return {
                    "content": record["local_data"]["node"].get("content", ""),
                    "sources": ["blockchain"],
                    "credibility": 0.98,  # Blockchain-verified high credibility
                    "verification_hash": record["blockchain_record"].get("data_hash", ""),
                    "blockchain_verified": True
                }
        except Exception as e:
            print(f"Blockchain check failed: {e}")
        
        return None

    def get_search_analytics(self) -> Dict:
        """Get analytics about search patterns and credibility."""
        return {
            "total_queries": len(self.query_history),
            "average_credibility_threshold": sum(q["min_credibility"] for q in self.query_history) / len(self.query_history) if self.query_history else 0,
            "most_common_queries": self._get_common_queries(),
            "cache_hit_rate": len(self.cache) / max(len(self.query_history), 1)
        }

    def _get_common_queries(self) -> List[Dict]:
        """Get most common queries."""
        from collections import Counter
        query_counts = Counter(q["query"] for q in self.query_history)
        return [{"query": q, "count": c} for q, c in query_counts.most_common(5)]