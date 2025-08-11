#!/usr/bin/env python3
"""
Script to run demo queries against the Neo4j database.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from retrieval.hybrid_retriever import HybridRetriever
from config.settings import get_settings

def main():
    """Main function to run demo queries."""
    try:
        settings = get_settings()
        retriever = HybridRetriever(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password
        )
        # set weights
        retriever.set_weights(vector_weight=0.6, fulltext_weight=0.4, semantic_weight=0.1)
        # Example queries
        queries = [
            "What is machine learning?",
            "Type 2 Diabetes"
        ]
        
        print("Running demo queries...")
        for query in queries:
            print(f"\nQuery: {query}")
            results = retriever.retrieve(query, top_k=3)
            print(f"Results: {len(results)} found")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result}")
            # explain the retrieval
            print(f"Explaining retrieval...")
            explanation = retriever.explain_retrieval(query, top_k=3)
            print(f"Explanation: {explanation}")
        
    except Exception as e:
        print(f"Error during demo queries: {e}")
        sys.exit(1)
    finally:
        if 'retriever' in locals():
            retriever.close()

if __name__ == "__main__":
    main()
