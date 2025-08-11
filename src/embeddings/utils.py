"""
Utility functions for embedding operations.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
import json

def normalize_vector(vector: List[float]) -> List[float]:
    """Normalize a vector to unit length."""
    vector = np.array(vector)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector.tolist()
    return (vector / norm).tolist()

def batch_embeddings(embeddings: List[List[float]], batch_size: int = 100) -> List[List[List[float]]]:
    """Split embeddings into batches."""
    return [embeddings[i:i + batch_size] for i in range(0, len(embeddings), batch_size)]

def save_embeddings(embeddings: Dict[str, List[float]], filepath: str):
    """Save embeddings to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(embeddings, f, indent=2)

def load_embeddings(filepath: str) -> Dict[str, List[float]]:
    """Load embeddings from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def compute_embedding_statistics(embeddings: List[List[float]]) -> Dict[str, float]:
    """Compute statistics for a collection of embeddings."""
    if not embeddings:
        return {}
    
    # Convert to numpy array for efficient computation
    embeddings_array = np.array(embeddings)
    
    stats = {
        'count': len(embeddings),
        'dimension': embeddings_array.shape[1] if embeddings_array.size > 0 else 0,
        'mean_norm': float(np.mean([np.linalg.norm(emb) for emb in embeddings_array])),
        'std_norm': float(np.std([np.linalg.norm(emb) for emb in embeddings_array])),
        'min_norm': float(np.min([np.linalg.norm(emb) for emb in embeddings_array])),
        'max_norm': float(np.max([np.linalg.norm(emb) for emb in embeddings_array]))
    }
    
    return stats

def find_similar_embeddings(
    query_embedding: List[float], 
    candidate_embeddings: List[List[float]], 
    top_k: int = 5,
    threshold: float = 0.0
) -> List[Tuple[int, float]]:
    """Find most similar embeddings to a query embedding."""
    similarities = []
    
    for i, candidate in enumerate(candidate_embeddings):
        similarity = cosine_similarity(query_embedding, candidate)
        if similarity >= threshold:
            similarities.append((i, similarity))
    
    # Sort by similarity (descending) and return top_k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))
