"""
Neo4j fulltext retriever implementation for the hybridRAG system.
"""

from typing import List, Tuple, Dict, Any
from neo4j import GraphDatabase
from config.settings import get_settings

class Neo4jFulltextRetriever:
    """Neo4j fulltext retriever for text-based search."""
    
    def __init__(self, uri: str = None, user: str = None, password: str = None):
        """Initialize the Neo4j fulltext retriever."""
        settings = get_settings()
        self.uri = uri or settings.neo4j_uri
        self.user = user or settings.neo4j_user
        self.password = password or settings.neo4j_password
        
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
    
    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()
    
    def create_fulltext_index(self, index_name: str, label: str, properties: List[str]):
        """Create a fulltext index on node properties."""
        with self.driver.session() as session:
            properties_str = ', '.join([f'n.{prop}' for prop in properties])
            query = f"""
            CALL db.index.fulltext.createNodeIndex(
                '{index_name}',
                '{label}',
                [{properties_str}]
            )
            """
            try:
                session.run(query)
                print(f"Fulltext index '{index_name}' created successfully")
            except Exception as e:
                print(f"Index creation failed (may already exist): {e}")
    
    def fulltext_search(self, query: str, index_name: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Perform fulltext search using the specified index."""
        with self.driver.session() as session:
            cypher_query = f"""
            CALL db.index.fulltext.queryNodes(
                '{index_name}',
                $search_text
            )
            YIELD node, score
            RETURN node, score
            ORDER BY score DESC
            LIMIT {top_k}
            """
            
            result = session.run(cypher_query, search_text=query)
            return [(record["node"], record["score"]) for record in result]
    
    def fuzzy_search(self, query: str, label: str, property: str, top_k: int = 5, 
                    similarity_threshold: float = 0.7) -> List[Tuple[Dict, float]]:
        """Perform fuzzy search using string similarity."""
        with self.driver.session() as session:
            cypher_query = f"""
            MATCH (n:{label})
            WITH n, apoc.text.fuzzyMatch(n.{property}, $search_text) as similarity
            WHERE similarity >= $threshold
            RETURN n, similarity as score
            ORDER BY similarity DESC
            LIMIT {top_k}
            """
            
            result = session.run(cypher_query, search_text=query, threshold=similarity_threshold)
            return [(record["n"], record["score"]) for record in result]
    
    def regex_search(self, pattern: str, label: str, property: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Perform regex search on text properties."""
        with self.driver.session() as session:
            query_cypher = f"""
            MATCH (n:{label})
            WHERE n.{property} =~ $pattern
            RETURN n, 1.0 as score
            LIMIT {top_k}
            """
            
            result = session.run(query_cypher, pattern=pattern)
            return [(record["n"], record["score"]) for record in result]
    
    def phrase_search(self, phrase: str, label: str, property: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Search for exact phrases in text."""
        with self.driver.session() as session:
            query_cypher = f"""
            MATCH (n:{label})
            WHERE n.{property} CONTAINS $phrase
            RETURN n, 1.0 as score
            ORDER BY size(n.{property}) ASC
            LIMIT {top_k}
            """
            
            result = session.run(query_cypher, phrase=phrase)
            return [(record["n"], record["score"]) for record in result]
    
    def semantic_search(self, query: str, label: str, property: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Perform semantic search using text analysis."""
        with self.driver.session() as session:
            # This is a simplified semantic search - you might want to implement
            # more sophisticated text analysis
            cypher_query = f"""
            MATCH (n:{label})
            WITH n, 
                 size([word IN split(toLower(n.{property}), ' ') 
                       WHERE word IN split(toLower($search_text), ' ')]) as common_words,
                 size(split(toLower(n.{property}), ' ')) as total_words
            WHERE common_words > 0
            RETURN n, 
                   (common_words * 1.0 / total_words) as score
            ORDER BY score DESC
            LIMIT {top_k}
            """
            
            result = session.run(cypher_query, search_text=query)
            return [(record["n"], record["score"]) for record in result]
