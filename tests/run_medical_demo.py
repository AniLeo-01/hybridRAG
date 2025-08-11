import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm.pipeline import LLMPipeline
from config.settings import get_settings

def main():
    """Main function to run medical-focused demo queries."""
    try:
        settings = get_settings()
        pipeline = LLMPipeline(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password
        )
        
        # Medical queries that should match the dataset content
        queries = [
            "What are calcium channel blockers?",
            "How do calcium channel blockers affect Type 2 Diabetes?",
            "What is metformin used for?",
            "How does metformin help with asthma?",
            "What are statins?",
            "How do statins help with hypertension?"
        ]
        
        print("Medical HybridRAG Demo")
        print("=" * 50)
        print("Testing queries relevant to the medical dataset...")
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
                
                if result['retrieved_documents'] > 0:
                    print(f"\nResponse:\n{result['response']}")
                else:
                    print("No relevant documents found.")
                
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
        print(f"Error during medical demo: {e}")
        sys.exit(1)
    finally:
        if 'pipeline' in locals():
            pipeline.close()

if __name__ == "__main__":
    main()
