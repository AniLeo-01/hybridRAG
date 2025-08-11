import numpy as np
from typing import List, Union
from openai import OpenAI
from config.settings import get_settings

class EmbeddingGenerator:
    """Generate embeddings for text using various models."""
    
    def __init__(self, model_name: str = "text-embedding-ada-002"):
        """Initialize the embedding generator."""
        self.model_name = model_name
        self.settings = get_settings()
        
        if self.settings.openai_api_key:
            self.client = OpenAI(api_key=self.settings.openai_api_key)
        else:
            self.client = None
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if not self.client:
            # Fallback to random embedding for testing
            return self._generate_random_embedding(text)
        
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model_name
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating OpenAI embedding: {e}")
            return self._generate_random_embedding(text)
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return [self.generate_embedding(text) for text in texts]
    
    def _generate_random_embedding(self, text: str, dimension: int = 1536) -> List[float]:
        """Generate a random embedding for testing purposes."""
        # Use text hash as seed for reproducible random embeddings
        seed = hash(text) % (2**32)
        np.random.seed(seed)
        return np.random.normal(0, 1, dimension).tolist()
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
