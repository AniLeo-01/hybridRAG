"""
Hybrid retriever implementation combining multiple search strategies.
"""

from typing import List, Tuple, Dict, Any, Optional
from config.settings import get_settings
from .neo4j_vector_store import Neo4jVectorStore
from .neo4j_fulltext_retriever import Neo4jFulltextRetriever

class HybridRetriever:
    """Hybrid retriever that combines vector, fulltext, and semantic search."""
    
    def __init__(self, uri: str = None, user: str = None, password: str = None):
        """Initialize the hybrid retriever."""
        settings = get_settings()
        self.uri = uri or settings.neo4j_uri
        self.user = user or settings.neo4j_user
        self.password = password or settings.neo4j_password
        
        self.vector_store = Neo4jVectorStore(self.uri, self.user, self.password)
        self.fulltext_retriever = Neo4jFulltextRetriever(self.uri, self.user, self.password)
        
        # Default weights for different search strategies
        self.vector_weight = 0.4
        self.fulltext_weight = 0.3
        self.semantic_weight = 0.3
    
    def close(self):
        """Close all database connections."""
        self.vector_store.close()
        self.fulltext_retriever.close()
    
    def set_weights(self, vector_weight: float = None, fulltext_weight: float = None, 
                   semantic_weight: float = None):
        """Set weights for different search strategies."""
        if vector_weight is not None:
            self.vector_weight = vector_weight
        if fulltext_weight is not None:
            self.fulltext_weight = fulltext_weight
        if semantic_weight is not None:
            self.semantic_weight = semantic_weight
        
        # Normalize weights to sum to 1.0
        total_weight = self.vector_weight + self.fulltext_weight + self.semantic_weight
        self.vector_weight /= total_weight
        self.fulltext_weight /= total_weight
        self.semantic_weight /= total_weight
    
    def retrieve(self, query: str, top_k: int = 5, strategy: str = "hybrid") -> List[Tuple[Dict, float]]:
        """Retrieve documents using the specified strategy."""
        if strategy == "vector":
            return self.vector_store.vector_search(query, top_k)
        elif strategy == "fulltext":
            return self.fulltext_retriever.fulltext_search(query, "documentFulltextIndex", top_k)
        elif strategy == "semantic":
            return self.fulltext_retriever.semantic_search(query, "Document", "text", top_k)
        elif strategy == "hybrid":
            return self._hybrid_retrieve(query, top_k)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _hybrid_retrieve(self, query: str, top_k: int) -> List[Tuple[Dict, float]]:
        """Perform hybrid retrieval combining multiple strategies."""
        # Get results from each strategy
        vector_results = self.vector_store.vector_search(query, top_k * 2)
        fulltext_results = self.fulltext_retriever.fulltext_search(query, "documentFulltextIndex", top_k * 2)
        semantic_results = self.fulltext_retriever.semantic_search(query, "Document", "text", top_k * 2)
        
        # Combine and rerank results
        combined_results = self._combine_and_rerank(
            vector_results, fulltext_results, semantic_results, top_k
        )
        
        return combined_results
    
    def _combine_and_rerank(self, vector_results: List[Tuple[Dict, float]], 
                           fulltext_results: List[Tuple[Dict, float]],
                           semantic_results: List[Tuple[Dict, float]], 
                           top_k: int) -> List[Tuple[Dict, float]]:
        """Combine results from different strategies and rerank them."""
        # Create a dictionary to store combined scores
        combined_scores = {}
        
        # Add vector search scores
        for node, score in vector_results:
            node_id = node.id
            combined_scores[node_id] = {
                'node': node,
                'vector_score': score,
                'fulltext_score': 0.0,
                'semantic_score': 0.0,
                'combined_score': score * self.vector_weight
            }
        
        # Add fulltext search scores
        for node, score in fulltext_results:
            node_id = node.id
            if node_id in combined_scores:
                combined_scores[node_id]['fulltext_score'] = score
                combined_scores[node_id]['combined_score'] += score * self.fulltext_weight
            else:
                combined_scores[node_id] = {
                    'node': node,
                    'vector_score': 0.0,
                    'fulltext_score': score,
                    'semantic_score': 0.0,
                    'combined_score': score * self.fulltext_weight
                }
        
        # Add semantic search scores
        for node, score in semantic_results:
            node_id = node.id
            if node_id in combined_scores:
                combined_scores[node_id]['semantic_score'] = score
                combined_scores[node_id]['combined_score'] += score * self.semantic_weight
            else:
                combined_scores[node_id] = {
                    'node': node,
                    'vector_score': 0.0,
                    'fulltext_score': 0.0,
                    'semantic_score': score,
                    'combined_score': score * self.semantic_weight
                }
        
        # Sort by combined score and return top_k
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )
        
        return [(item['node'], item['combined_score']) for item in sorted_results[:top_k]]
    
    def explain_retrieval(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Explain the retrieval process and show scores from each strategy."""
        # Get results from each strategy
        vector_results = self.vector_store.vector_search(query, top_k)
        fulltext_results = self.fulltext_retriever.fulltext_search(query, "documentFulltextIndex", top_k)
        semantic_results = self.fulltext_retriever.semantic_search(query, "Document", "text", top_k)
        
        # Get hybrid results
        hybrid_results = self._hybrid_retrieve(query, top_k)
        
        explanation = {
            'query': query,
            'strategy_weights': {
                'vector': self.vector_weight,
                'fulltext': self.fulltext_weight,
                'semantic': self.semantic_weight
            },
            'vector_results': vector_results,
            'fulltext_results': fulltext_results,
            'semantic_results': semantic_results,
            'hybrid_results': hybrid_results,
            'total_results': len(hybrid_results)
        }
        
        return explanation
    
    def create_indexes(self):
        """Create all necessary indexes for the hybrid retriever."""
        print("Creating vector index...")
        self.vector_store.create_vector_index(
            "documentEmbeddingIndex", "Document", "embedding", 1536
        )
        
        print("Creating fulltext index...")
        self.fulltext_retriever.create_fulltext_index(
            "documentFulltextIndex", "Document", ["text"]
        )
        
        print("All indexes created successfully!")
