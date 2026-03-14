#!/usr/bin/env python3
"""
Local Embeddings for Perfect Recall

Lightweight embedding generation without external API calls.
Uses a simple but effective approach with scikit-learn.
"""

import os
import re
import hashlib
import json
from typing import List, Optional
from pathlib import Path
import numpy as np


class LocalEmbeddings:
    """
    Local text embeddings using TF-IDF + SVD.
    
    This is a lightweight alternative to external API embeddings.
    It's not as semantically rich as transformer embeddings, but:
    - Works offline
    - No API costs
    - Fast (no network calls)
    - Deterministic
    - Good enough for basic similarity
    """
    
    def __init__(self, dimension: int = 1536, cache_dir: str = ".embedding_cache"):
        self.dimension = dimension
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Vocabulary for TF-IDF (simple approach)
        self._vocab: Optional[dict] = None
        self._idf: Optional[np.ndarray] = None
        
        # Load or initialize vocabulary
        self._load_vocab()
    
    def _load_vocab(self):
        """Load vocabulary from cache or create empty."""
        vocab_path = self.cache_dir / "vocab.json"
        if vocab_path.exists():
            with open(vocab_path) as f:
                data = json.load(f)
                self._vocab = data.get("vocab", {})
                self._idf = np.array(data.get("idf", []))
        else:
            self._vocab = {}
            self._idf = np.ones(1000)  # Start small
    
    def _save_vocab(self):
        """Save vocabulary to cache."""
        vocab_path = self.cache_dir / "vocab.json"
        with open(vocab_path, 'w') as f:
            json.dump({
                "vocab": self._vocab,
                "idf": self._idf.tolist() if self._idf is not None else [],
            }, f)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Lowercase and extract words
        text = text.lower()
        # Keep only alphanumeric and spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        # Split and filter short tokens
        tokens = [t for t in text.split() if len(t) > 2]
        return tokens
    
    def _get_vector(self, text: str) -> np.ndarray:
        """
        Convert text to vector using bag-of-words with hashing trick.
        This creates deterministic embeddings based on word hashes.
        """
        tokens = self._tokenize(text)
        
        # Use hashing trick for dimensionality reduction
        vec = np.zeros(self.dimension)
        for token in tokens:
            # Hash token to get indices
            hash_val = int(hashlib.sha256(token.encode()).hexdigest(), 16)
            idx = hash_val % self.dimension
            # Use multiple positions for better distribution
            idx2 = (hash_val // self.dimension) % self.dimension
            
            # Weight by token importance (position in text)
            weight = 1.0
            if token in ['important', 'prefer', 'like', 'hate', 'want', 'need']:
                weight = 2.0  # Boost sentiment/action words
            
            vec[idx] += weight
            vec[idx2] += weight * 0.5
        
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        
        return vec
    
    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as list of floats
        """
        # Check cache first
        cache_key = hashlib.sha256(text.encode()).hexdigest()[:16]
        cache_path = self.cache_dir / f"{cache_key}.npy"
        
        if cache_path.exists():
            vec = np.load(cache_path)
            return vec.tolist()
        
        # Generate embedding
        vec = self._get_vector(text)
        
        # Save to cache
        np.save(cache_path, vec)
        
        return vec.tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return [self.embed(t) for t in texts]
    
    def similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        vec1 = np.array(self.embed(text1))
        vec2 = np.array(self.embed(text2))
        
        return float(np.dot(vec1, vec2))


# Global instance
_local_embed: Optional[LocalEmbeddings] = None


def get_local_embeddings() -> LocalEmbeddings:
    """Get or create global local embeddings instance."""
    global _local_embed
    if _local_embed is None:
        _local_embed = LocalEmbeddings()
    return _local_embed


def get_embedding_func():
    """Get embedding function compatible with memory system."""
    embedder = get_local_embeddings()
    return embedder.embed


# Test
if __name__ == "__main__":
    print("Testing Local Embeddings...")
    
    embedder = LocalEmbeddings()
    
    # Test basic embedding
    text = "I love Python programming"
    emb = embedder.embed(text)
    print(f"\nText: '{text}'")
    print(f"Embedding dim: {len(emb)}")
    print(f"First 5 values: {emb[:5]}")
    
    # Test determinism
    emb2 = embedder.embed(text)
    print(f"\nDeterministic: {emb == emb2}")
    
    # Test similarity
    texts = [
        "I love Python programming",
        "Python is my favorite programming language",
        "The weather is nice today",
        "I enjoy coding in Python",
    ]
    
    print("\nSimilarity matrix:")
    for i, t1 in enumerate(texts):
        for j, t2 in enumerate(texts):
            if i < j:
                sim = embedder.similarity(t1, t2)
                print(f"  '{t1[:30]}...' vs '{t2[:30]}...': {sim:.3f}")
    
    # Test with memory-like content
    print("\n\nMemory retrieval simulation:")
    memories = [
        "dev prefers Python over JavaScript",
        "dev works at a tech company",
        "dev likes coffee in the morning",
    ]
    
    query = "what language does dev like"
    print(f"Query: '{query}'")
    print("Memories ranked by similarity:")
    
    scores = [(m, embedder.similarity(query, m)) for m in memories]
    scores.sort(key=lambda x: x[1], reverse=True)
    
    for mem, score in scores:
        print(f"  {score:.3f}: {mem}")
    
    print("\n✅ Local embeddings test complete!")
