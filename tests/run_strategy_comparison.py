import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm.pipeline import LLMPipeline
from config.settings import get_settings

def main():
    """Main function to compare different retrieval strategies."""
    try:
        settings = get_settings()
        pipeline = LLMPipeline(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password
        )
        
        # Test query
        query = "Metformin asthma management"
        
        print("HybridRAG Strategy Comparison Demo")
        print("=" * 60)
        print(f"Query: {query}")
        print("=" * 60)
        
        # Test different strategies
        strategies = ["vector", "fulltext", "semantic", "hybrid"]
        
        for strategy in strategies:
            print(f"\n{strategy.upper()} STRATEGY")
            print("-" * 40)
            
            try:
                result = pipeline.process_query(query, strategy=strategy, top_k=3, include_explanation=True)
                
                if 'error' in result:
                    print(f"Error: {result['error']}")
                else:
                    print(f"Retrieved documents: {result['retrieved_documents']}")
                    print(f"Context length: {result['context_length']} characters")
                    
                    # Show explanation if available
                    if 'explanation' in result:
                        explanation = result['explanation']
                        print(f"Strategy weights: {explanation['strategy_weights']}")
                    
                    print(f"\nResponse preview: {result['response'][:200]}...")
                    
            except Exception as e:
                print(f"Strategy failed: {e}")
        
        # Show pipeline configuration
        print("\n" + "=" * 60)
        print("Pipeline Configuration:")
        stats = pipeline.get_pipeline_stats()
        print(f"  OpenAI configured: {stats['openai_configured']}")
        print(f"  Max context length: {stats['max_context_length']}")
        print(f"  Temperature: {stats['temperature']}")
        print(f"  Max tokens: {stats['max_tokens']}")
        print(f"  Default retriever weights: {stats['retriever_weights']}")
        
    except Exception as e:
        print(f"Error during strategy comparison: {e}")
        sys.exit(1)
    finally:
        if 'pipeline' in locals():
            pipeline.close()

if __name__ == "__main__":
    main()
