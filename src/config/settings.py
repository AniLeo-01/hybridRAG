"""
Configuration settings for the hybridRAG system.
"""

import os
import yaml
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings."""
    
    # Neo4j Configuration
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    
    # Retrieval Configuration
    default_top_k: int = 5
    vector_similarity_threshold: float = 0.7
    strategy_weights: dict = {"vector": 0.4, "fulltext": 0.3, "semantic": 0.3}
    
    # Embedding Configuration
    embedding_model: str = "text-embedding-ada-002"
    embedding_dimension: int = 1536
    embedding_batch_size: int = 100
    
    # LLM Configuration
    max_context_length: int = 4000
    temperature: float = 0.7
    max_tokens: int = 500
    
    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None
    
    # Index Configuration
    vector_index_name: str = "documentEmbeddingIndex"
    vector_index_label: str = "Document"
    vector_index_property: str = "embedding"
    vector_index_dimension: int = 1536
    fulltext_index_name: str = "documentFulltextIndex"
    fulltext_index_label: str = "Document"
    fulltext_index_properties: list = ["text"]
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow"
    
    def __init__(self, **kwargs):
        # Initialize with environment variables first
        super().__init__(**kwargs)
        
        # Then load and override with config.yaml
        self._load_yaml_config()
    
    def _load_yaml_config(self):
        """Load configuration from config.yaml file."""
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                if config_data:
                    # Load database configuration
                    if 'database' in config_data and 'neo4j' in config_data['database']:
                        neo4j_config = config_data['database']['neo4j']
                        self.neo4j_uri = neo4j_config.get('uri', self.neo4j_uri)
                        self.neo4j_user = neo4j_config.get('user', self.neo4j_user)
                        self.neo4j_password = neo4j_config.get('password', self.neo4j_password)
                        self.neo4j_database = neo4j_config.get('database', self.neo4j_database)
                    
                    # Load retrieval configuration
                    if 'retrieval' in config_data:
                        retrieval_config = config_data['retrieval']
                        self.default_top_k = retrieval_config.get('default_top_k', self.default_top_k)
                        self.vector_similarity_threshold = retrieval_config.get('vector_similarity_threshold', self.vector_similarity_threshold)
                        if 'strategy_weights' in retrieval_config:
                            self.strategy_weights = retrieval_config['strategy_weights']
                    
                    # Load embedding configuration
                    if 'embeddings' in config_data:
                        embedding_config = config_data['embeddings']
                        self.embedding_model = embedding_config.get('model', self.embedding_model)
                        self.embedding_dimension = embedding_config.get('dimension', self.embedding_dimension)
                        self.embedding_batch_size = embedding_config.get('batch_size', self.embedding_batch_size)
                    
                    # Load LLM configuration
                    if 'llm' in config_data:
                        llm_config = config_data['llm']
                        self.max_context_length = llm_config.get('max_context_length', self.max_context_length)
                        self.temperature = llm_config.get('temperature', self.temperature)
                        self.max_tokens = llm_config.get('max_tokens', self.max_tokens)
                    
                    # Load logging configuration
                    if 'logging' in config_data:
                        logging_config = config_data['logging']
                        self.log_level = logging_config.get('level', self.log_level)
                        self.log_format = logging_config.get('format', self.log_format)
                        self.log_file = logging_config.get('file', self.log_file)
                    
                    # Load index configuration
                    if 'indexes' in config_data:
                        if 'vector' in config_data['indexes']:
                            vector_config = config_data['indexes']['vector']
                            self.vector_index_name = vector_config.get('name', self.vector_index_name)
                            self.vector_index_label = vector_config.get('label', self.vector_index_label)
                            self.vector_index_property = vector_config.get('property', self.vector_index_property)
                            self.vector_index_dimension = vector_config.get('dimension', self.vector_index_dimension)
                        
                        if 'fulltext' in config_data['indexes']:
                            fulltext_config = config_data['indexes']['fulltext']
                            self.fulltext_index_name = fulltext_config.get('name', self.fulltext_index_name)
                            self.fulltext_index_label = fulltext_config.get('label', self.fulltext_index_label)
                            self.fulltext_index_properties = fulltext_config.get('properties', self.fulltext_index_properties)
                
                # print(f"✅ Configuration loaded from {config_path}")
                
            except Exception as e:
                print(f"⚠️ Warning: Could not load config.yaml: {e}")
                print("   Using default configuration values")
        else:
            print(f"⚠️ Warning: config.yaml not found at {config_path}")
            print("   Using default configuration values")

def get_settings() -> Settings:
    """Get application settings."""
    return Settings()

# Global settings instance
settings = get_settings()
