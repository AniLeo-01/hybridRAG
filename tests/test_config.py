#!/usr/bin/env python3
"""
Test script to verify the new configuration system.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_configuration():
    """Test the new configuration system."""
    try:
        print("Testing New Configuration System...\n")
        
        # Import and test settings
        from config.settings import get_settings
        settings = get_settings()
        
        print("Configuration loaded successfully!")
        print("\nConfiguration Values:")
        print("=" * 50)
        
        # Database Configuration
        print("Database Configuration:")
        print(f"   Neo4j URI: {settings.neo4j_uri}")
        print(f"   Neo4j User: {settings.neo4j_user}")
        print(f"   Neo4j Database: {settings.neo4j_database}")
        print(f"   Neo4j Password: {'*' * len(settings.neo4j_password)}")
        
        # Retrieval Configuration
        print("\nRetrieval Configuration:")
        print(f"   Default Top-K: {settings.default_top_k}")
        print(f"   Vector Similarity Threshold: {settings.vector_similarity_threshold}")
        print(f"   Strategy Weights: {settings.strategy_weights}")
        
        # Embedding Configuration
        print("\nEmbedding Configuration:")
        print(f"   Model: {settings.embedding_model}")
        print(f"   Dimension: {settings.embedding_dimension}")
        print(f"   Batch Size: {settings.embedding_batch_size}")
        
        # LLM Configuration
        print("\nLLM Configuration:")
        print(f"   Max Context Length: {settings.max_context_length}")
        print(f"   Temperature: {settings.temperature}")
        print(f"   Max Tokens: {settings.max_tokens}")
        
        # Logging Configuration
        print("\nLogging Configuration:")
        print(f"   Level: {settings.log_level}")
        print(f"   Format: {settings.log_format}")
        print(f"   File: {settings.log_file}")
        
        # Index Configuration
        print("\nIndex Configuration:")
        print(f"   Vector Index: {settings.vector_index_name}")
        print(f"   Vector Label: {settings.vector_index_label}")
        print(f"   Vector Property: {settings.vector_index_property}")
        print(f"   Vector Dimension: {settings.vector_index_dimension}")
        print(f"   Fulltext Index: {settings.fulltext_index_name}")
        print(f"   Fulltext Properties: {settings.fulltext_index_properties}")
        
        print("\nConfiguration test completed successfully!")
        print("\nThe system is now using:")
        print("   • config.yaml as PRIMARY configuration")
        print("   • .env for sensitive overrides (optional)")
        print("   • Environment variables can still override both")
        
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_configuration()
    sys.exit(0 if success else 1)
