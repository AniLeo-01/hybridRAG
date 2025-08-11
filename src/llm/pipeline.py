from typing import List, Dict, Any, Optional
from config.settings import get_settings
from retrieval.hybrid_retriever import HybridRetriever
from embeddings.generator import EmbeddingGenerator
from openai import OpenAI

class LLMPipeline:
    """LLM pipeline that orchestrates retrieval and generation."""
    
    def __init__(self, uri: str = None, user: str = None, password: str = None):
        """Initialize the LLM pipeline."""
        settings = get_settings()
        self.uri = uri or settings.neo4j_uri
        self.user = user or settings.neo4j_user
        self.password = password or settings.neo4j_password
        
        self.retriever = HybridRetriever(self.uri, self.user, self.password)
        self.embedding_generator = EmbeddingGenerator()
        
        # Pipeline configuration
        self.max_context_length = settings.max_context_length
        self.temperature = settings.temperature
        self.max_tokens = settings.max_tokens
        
        # Initialize OpenAI client if API key is available
        if settings.openai_api_key:
            self.openai_client = OpenAI(api_key=settings.openai_api_key)
        else:
            self.openai_client = None
    
    def close(self):
        self.retriever.close()
    
    def process_query(self, query: str, strategy: str = "hybrid", 
                     top_k: int = 5, include_explanation: bool = False) -> Dict[str, Any]:
        """Process a query through the complete pipeline."""
        try:
            # Step 1: Retrieve relevant documents
            retrieved_docs = self.retriever.retrieve(query, top_k, strategy)
            
            # Step 2: Prepare context from retrieved documents
            context = self._prepare_context(retrieved_docs)
            
            # Step 3: Generate response using LLM
            response = self._generate_response(query, context)
            
            # Step 4: Prepare result
            result = {
                'query': query,
                'strategy': strategy,
                'retrieved_documents': len(retrieved_docs),
                'response': response,
                'context_length': len(context)
            }
            
            if include_explanation:
                result['explanation'] = self.retriever.explain_retrieval(query, top_k)
            
            return result
            
        except Exception as e:
            return {
                'error': str(e),
                'query': query,
                'strategy': strategy
            }
    
    def _prepare_context(self, retrieved_docs: List[tuple]) -> str:
        """Prepare context string from retrieved documents."""
        context_parts = []
        
        for i, (node, score) in enumerate(retrieved_docs):
            # Extract text content from the node
            if hasattr(node, 'text'):
                text = node.text
            elif hasattr(node, 'title'):
                text = f"Title: {node.title}\n"
                if hasattr(node, 'source'):
                    text += f"Source: {node.source}\n"
            else:
                text = str(node)
            
            context_parts.append(f"Document {i+1} (Score: {score:.3f}):\n{text}\n")
        
        context = "\n".join(context_parts)
        
        # Truncate if too long
        if len(context) > self.max_context_length:
            context = context[:self.max_context_length] + "..."
        
        return context
    
    def _generate_response(self, query: str, context: str) -> str:
        """Generate response using the context and an LLM."""
        if not self.openai_client:
            # Fallback response when OpenAI is not configured
            return f"Based on the retrieved documents, here's what I found about '{query}':\n\n{context}\n\n[Note: OpenAI API key not configured. This is a fallback response.]"
        
        try:
            # Create system prompt for RAG
            system_prompt = """You are a helpful AI assistant that answers questions based on the provided context. 
            Use only the information from the context to answer the question. If the context doesn't contain 
            enough information to answer the question, say so. Be concise but informative."""
            
            # Create user prompt with context
            user_prompt = f"""Question: {query}

            Context:
            {context}

            Please provide a clear and accurate answer based on the context above:"""
            
            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            # Fallback response on error
            return f"Based on the retrieved documents, here's what I found about '{query}':\n\n{context}\n\n[Error generating LLM response: {str(e)}]"
    
    def batch_process(self, queries: List[str], strategy: str = "hybrid", 
                     top_k: int = 5) -> List[Dict[str, Any]]:
        """Process multiple queries in batch."""
        results = []
        
        for query in queries:
            result = self.process_query(query, strategy, top_k)
            results.append(result)
        
        return results
    
    def set_pipeline_config(self, max_context_length: int = None, 
                           temperature: float = None, max_tokens: int = None):
        """Set pipeline configuration parameters."""
        if max_context_length is not None:
            self.max_context_length = max_context_length
        if temperature is not None:
            self.temperature = temperature
        if max_tokens is not None:
            self.max_tokens = max_tokens
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics and configuration."""
        return {
            'max_context_length': self.max_context_length,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'retriever_weights': {
                'vector': self.retriever.vector_weight,
                'fulltext': self.retriever.fulltext_weight,
                'semantic': self.retriever.semantic_weight
            },
            'openai_configured': self.openai_client is not None
        }
