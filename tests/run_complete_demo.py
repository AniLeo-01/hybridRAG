import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm.pipeline import LLMPipeline
from config.settings import get_settings

def main():
    """Main function to demonstrate complete hybridRAG capabilities."""
    try:
        settings = get_settings()
        pipeline = LLMPipeline(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password
        )
        
        print("Complete HybridRAG System Demo")
        print("=" * 60)
        
        # Test 1: Basic hybrid retrieval
        print("\nTest 1: Basic Hybrid Retrieval")
        print("-" * 40)
        query = "What are calcium channel blockers used for?"
        result = pipeline.process_query(query, strategy="hybrid", top_k=3, include_explanation=True)
        
        if 'error' not in result:
            print(f"Query: {query}")
            print(f"Retrieved {result['retrieved_documents']} documents")
            print(f"Response: {result['response'][:150]}...")
        else:
            print(f"Error: {result['error']}")
        
        # Test 2: Strategy comparison
        print("\nTest 2: Strategy Comparison")
        print("-" * 40)
        strategies = ["vector", "fulltext", "semantic", "hybrid"]
        query = "metformin asthma management"
        
        for strategy in strategies:
            result = pipeline.process_query(query, strategy=strategy, top_k=2)
            if 'error' not in result:
                print(f"{strategy.upper()}: {result['retrieved_documents']} docs, {result['context_length']} chars")
            else:
                print(f"{strategy.upper()}: {result['error']}")
        
        # Test 3: Custom weights
        print("\nTest 3: Custom Retrieval Weights")
        print("-" * 40)
        
        # Set custom weights
        pipeline.retriever.set_weights(vector_weight=0.7, fulltext_weight=0.2, semantic_weight=0.1)
        print("Set weights: Vector(0.7), Fulltext(0.2), Semantic(0.1)")
        
        result = pipeline.process_query("calcium channel blockers diabetes", strategy="hybrid", top_k=2)
        if 'error' not in result:
            print(f"Retrieved {result['retrieved_documents']} documents with custom weights")
        
        # Reset to default weights
        pipeline.retriever.set_weights(vector_weight=0.4, fulltext_weight=0.3, semantic_weight=0.3)
        print("Reset to default weights")
        
        # Test 4: Batch processing
        print("\nTest 4: Batch Processing")
        print("-" * 40)
        batch_queries = [
            "calcium channel blockers",
            "metformin benefits",
            "hypertension treatment"
        ]
        
        batch_results = pipeline.batch_process(batch_queries, strategy="hybrid", top_k=2)
        print(f"Processed {len(batch_results)} queries in batch")
        
        for i, result in enumerate(batch_results):
            if 'error' not in result:
                print(f"  Query {i+1}: {result['retrieved_documents']} docs, {result['context_length']} chars")
            else:
                print(f"  Query {i+1}: Error - {result['error']}")
        
        # Test 5: Pipeline configuration
        print("\nTest 5: Pipeline Configuration")
        print("-" * 40)
        
        # Show current stats
        stats = pipeline.get_pipeline_stats()
        print("Current Configuration:")
        print(f"  Max context length: {stats['max_context_length']}")
        print(f"  Temperature: {stats['temperature']}")
        print(f"  Max tokens: {stats['max_tokens']}")
        print(f"  Retriever weights: {stats['retriever_weights']}")
        print(f"  OpenAI configured: {stats['openai_configured']}")
        
        # Test 6: Context preparation
        print("\nTest 6: Context Preparation")
        print("-" * 40)
        
        # Test with a query that should return results
        result = pipeline.process_query("calcium channel blockers", strategy="fulltext", top_k=1)
        if 'error' not in result and result['retrieved_documents'] > 0:
            print(f"Context length: {result['context_length']} characters")
            print(f"Context preview: {result['response'][:100]}...")
        else:
            print("No context generated")
        
    except Exception as e:
        print(f"Error during complete demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if 'pipeline' in locals():
            pipeline.close()

if __name__ == "__main__":
    main()
