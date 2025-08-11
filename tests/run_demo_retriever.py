import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm.pipeline import LLMPipeline
from config.settings import get_settings

def main():
    """Main function to run demo queries through the complete pipeline."""
    try:
        settings = get_settings()
        pipeline = LLMPipeline(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password
        )
        
        # Example queries
        queries = [
            "What is machine learning?",
            "Explain neural networks",
            "How does deep learning work?",
            "What are the benefits of calcium channel blockers?",
            "How does metformin work?"
        ]
        
        print("Running hybridRAG pipeline demo...")
        print("=" * 50)
        
        for i, query in enumerate(queries, 1):
            print(f"\nQuery {i}: {query}")
            print("-" * 30)
            
            # Process query through the complete pipeline
            result = pipeline.process_query(query, strategy="hybrid", top_k=3, include_explanation=True)
            
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Retrieved {result['retrieved_documents']} documents")
                print(f"Context length: {result['context_length']} characters")
                print(f"\nResponse:\n{result['response']}")
                
                # Show explanation if available
                if 'explanation' in result:
                    explanation = result['explanation']
                    print(f"\nRetrieval Strategy Weights:")
                    print(f"  Vector: {explanation['strategy_weights']['vector']:.2f}")
                    print(f"  Fulltext: {explanation['strategy_weights']['fulltext']:.2f}")
                    print(f"  Semantic: {explanation['strategy_weights']['semantic']:.2f}")
        
        # Show pipeline statistics
        print("\n" + "=" * 50)
        print("Pipeline Statistics:")
        stats = pipeline.get_pipeline_stats()
        print(f"  OpenAI configured: {stats['openai_configured']}")
        print(f"  Max context length: {stats['max_context_length']}")
        print(f"  Temperature: {stats['temperature']}")
        print(f"  Max tokens: {stats['max_tokens']}")
        print(f"  Retriever weights: {stats['retriever_weights']}")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        sys.exit(1)
    finally:
        if 'pipeline' in locals():
            pipeline.close()

if __name__ == "__main__":
    main()
