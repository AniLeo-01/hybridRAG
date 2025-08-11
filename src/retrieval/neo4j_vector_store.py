"""
Neo4j vector store implementation for the hybridRAG system.
"""

from typing import List, Tuple, Dict, Any, Optional
from neo4j import GraphDatabase
from config.settings import get_settings
from embeddings.generator import EmbeddingGenerator

class Neo4jVectorStore:
    """Neo4j vector store for storing and retrieving vector embeddings."""
    
    def __init__(self, uri: str = None, user: str = None, password: str = None):
        """Initialize the Neo4j vector store."""
        settings = get_settings()
        self.uri = uri or settings.neo4j_uri
        self.user = user or settings.neo4j_user
        self.password = password or settings.neo4j_password
        
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self.embedding_generator = EmbeddingGenerator()
    
    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()
    
    def create_vector_index(self, index_name: str, label: str, property: str, dimension: int = 1536):
        """Create a vector index on a node property."""
        with self.driver.session() as session:
            query = f"""
            CALL db.index.vector.createNodeIndex(
                '{index_name}',
                '{label}',
                '{property}',
                {dimension},
                'cosine'
            )
            """
            try:
                session.run(query)
                print(f"Vector index '{index_name}' created successfully")
            except Exception as e:
                print(f"Index creation failed (may already exist): {e}")
    
    def store_embeddings(self, nodes: List[Dict[str, Any]], label: str, text_property: str = "text"):
        """Store nodes with their embeddings in Neo4j."""
        with self.driver.session() as session:
            for node in nodes:
                # Generate embedding for the text
                text = node.get(text_property, "")
                embedding = self.embedding_generator.generate_embedding(text)
                
                # Store node with embedding
                query = f"""
                CREATE (n:{label} {{
                    text: $text,
                    embedding: $embedding,
                    metadata: $metadata
                }})
                """
                
                metadata = {k: v for k, v in node.items() if k != text_property}
                session.run(query, text=text, embedding=embedding, metadata=metadata)
    
    def vector_search(self, query: str, top_k: int = 5, label: str = "Document", 
                     index_name: str = "documentEmbeddingIndex") -> List[Tuple[Dict, float]]:
        """Perform vector similarity search."""
        # Generate embedding for the query
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        with self.driver.session() as session:
            query_cypher = f"""
            CALL db.index.vector.queryNodes(
                '{index_name}',
                {top_k},
                $query_embedding
            )
            YIELD node, score
            RETURN node, score
            ORDER BY score DESC
            LIMIT {top_k}
            """
            
            result = session.run(query_cypher, query_embedding=query_embedding)
            return [(record["node"], record["score"]) for record in result]
    
    def hybrid_search(self, query: str, top_k: int = 5, label: str = "Document",
                     index_name: str = "documentEmbeddingIndex", 
                     fulltext_weight: float = 0.3,
                     vector_weight: float = 0.7) -> List[Tuple[Dict, float]]:
        """Perform hybrid search combining fulltext and vector search."""
        # Vector search
        vector_results = self.vector_search(query, top_k * 2, label, index_name)
        
        # Fulltext search (simplified - you might want to implement proper fulltext search)
        fulltext_results = self._simple_fulltext_search(query, top_k * 2, label)
        
        # Combine and rerank results
        combined_results = self._combine_results(
            vector_results, fulltext_results, fulltext_weight, vector_weight
        )
        
        return combined_results[:top_k]
    
    def _simple_fulltext_search(self, query: str, top_k: int, label: str) -> List[Tuple[Dict, float]]:
        """Simple fulltext search implementation."""
        with self.driver.session() as session:
            query_cypher = f"""
            MATCH (n:{label})
            WHERE n.text CONTAINS $query
            RETURN n, 0.5 as score
            LIMIT {top_k}
            """
            
            result = session.run(query_cypher, query=query)
            return [(record["n"], record["score"]) for record in result]
    
    def _combine_results(self, vector_results: List[Tuple[Dict, float]], 
                        fulltext_results: List[Tuple[Dict, float]],
                        fulltext_weight: float, vector_weight: float) -> List[Tuple[Dict, float]]:
        """Combine and rerank results from different search methods."""
        # Create a dictionary to store combined scores
        combined_scores = {}
        
        # Add vector search scores
        for node, score in vector_results:
            node_id = node.id
            combined_scores[node_id] = {
                'node': node,
                'vector_score': score,
                'fulltext_score': 0.0,
                'combined_score': score * vector_weight
            }
        
        # Add fulltext search scores
        for node, score in fulltext_results:
            node_id = node.id
            if node_id in combined_scores:
                combined_scores[node_id]['fulltext_score'] = score
                combined_scores[node_id]['combined_score'] += score * fulltext_weight
            else:
                combined_scores[node_id] = {
                    'node': node,
                    'vector_score': 0.0,
                    'fulltext_score': score,
                    'combined_score': score * fulltext_weight
                }
        
        # Sort by combined score and return
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )
        
        return [(item['node'], item['combined_score']) for item in sorted_results]
