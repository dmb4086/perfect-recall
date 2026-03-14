#!/usr/bin/env python3
"""
Voyage AI Embeddings Client

High-quality embeddings via Voyage AI API.
"""

import os
import asyncio
import hashlib
from typing import List, Optional
import aiohttp


class VoyageEmbeddings:
    """Client for Voyage AI embeddings API."""
    
    DEFAULT_MODEL = "voyage-3"
    DEFAULT_DIMENSION = 1024
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        input_type: str = "document",  # "document" or "query"
    ):
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY")
        self.model = model
        self.input_type = input_type
        self.base_url = "https://api.voyageai.com/v1"
        
        if not self.api_key:
            raise ValueError(
                "Voyage API key required. Set VOYAGE_API_KEY environment variable."
            )
    
    async def embed(
        self,
        texts: List[str],
        input_type: Optional[str] = None,
    ) -> List[List[float]]:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            input_type: "document" or "query" (overrides default)
            
        Returns:
            List of embedding vectors
        """
        if isinstance(texts, str):
            texts = [texts]
        
        itype = input_type or self.input_type
        
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
            payload = {
                "input": texts,
                "model": self.model,
                "input_type": itype,
            }
            
            async with session.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json=payload,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Voyage API error: {response.status} - {error_text}")
                
                data = await response.json()
                # Sort by index to maintain order
                embeddings = sorted(data["data"], key=lambda x: x["index"])
                return [e["embedding"] for e in embeddings]
    
    async def embed_one(self, text: str, input_type: Optional[str] = None) -> List[float]:
        """Embed a single text."""
        results = await self.embed([text], input_type=input_type)
        return results[0]


class CachedVoyageEmbeddings(VoyageEmbeddings):
    """Voyage embeddings with local file caching."""
    
    def __init__(self, cache_dir: str = ".embedding_cache", **kwargs):
        super().__init__(**kwargs)
        self.cache_dir = cache_dir
        self._memory_cache: dict = {}
        
        import os
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_path(self, key: str) -> str:
        """Get path for cache entry."""
        import os
        return os.path.join(self.cache_dir, f"voyage_{key}.json")
    
    def _get_cache_key(self, text: str, input_type: str) -> str:
        """Generate cache key from text and input type."""
        key_data = f"{self.model}:{input_type}:{text}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]
    
    async def embed_one(self, text: str, input_type: Optional[str] = None) -> List[float]:
        """Get embedding with caching."""
        itype = input_type or self.input_type
        cache_key = self._get_cache_key(text, itype)
        
        # Check memory cache
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]
        
        # Check disk cache
        cache_path = self._get_cache_path(cache_key)
        if os.path.exists(cache_path):
            import json
            with open(cache_path) as f:
                embedding = json.load(f)
            self._memory_cache[cache_key] = embedding
            return embedding
        
        # Fetch from API
        embedding = await super().embed_one(text, input_type=itype)
        
        # Save to caches
        self._memory_cache[cache_key] = embedding
        import json
        with open(cache_path, 'w') as f:
            json.dump(embedding, f)
        
        return embedding


# Global instance
_voyage_embed: Optional[CachedVoyageEmbeddings] = None


def get_voyage_embeddings() -> CachedVoyageEmbeddings:
    """Get or create global Voyage embeddings instance."""
    global _voyage_embed
    if _voyage_embed is None:
        _voyage_embed = CachedVoyageEmbeddings()
    return _voyage_embed


def get_embedding_func():
    """Get embedding function compatible with memory system."""
    embedder = get_voyage_embeddings()
    
    def embed_sync(text: str) -> List[float]:
        """Synchronous wrapper for async embed."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(embedder.embed_one(text))
            else:
                return loop.run_until_complete(embedder.embed_one(text))
        except RuntimeError:
            return asyncio.run(embedder.embed_one(text))
    
    return embed_sync


# Test
if __name__ == "__main__":
    async def test():
        # Check for API key
        api_key = os.getenv("VOYAGE_API_KEY")
        
        if not api_key:
            print("❌ No VOYAGE_API_KEY found in environment")
            return
        
        print("Testing Voyage embeddings...")
        
        client = CachedVoyageEmbeddings(api_key=api_key)
        
        # Single embed
        text = "Perfect Recall memory system"
        print(f"\nEmbedding: '{text}'")
        
        import time
        start = time.time()
        embedding = await client.embed_one(text)
        elapsed = time.time() - start
        
        print(f"  Dimension: {len(embedding)}")
        print(f"  First 5 values: {[round(x, 4) for x in embedding[:5]]}")
        print(f"  Latency: {elapsed:.3f}s")
        
        # Test caching
        print("\nTesting cache...")
        start = time.time()
        embedding2 = await client.embed_one(text)
        elapsed = time.time() - start
        print(f"  Cached lookup latency: {elapsed:.3f}s")
        print(f"  Same result: {embedding == embedding2}")
        
        # Test semantic similarity
        print("\nTesting semantic similarity...")
        texts = [
            "I love Python programming",
            "Python is my favorite language",
            "The weather is nice today",
        ]
        
        embeddings = await client.embed(texts)
        
        # Calculate cosine similarities
        import numpy as np
        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        sim_0_1 = cosine_sim(embeddings[0], embeddings[1])
        sim_0_2 = cosine_sim(embeddings[0], embeddings[2])
        
        print(f"  'Python' vs 'Python' similarity: {sim_0_1:.3f} (should be high, >0.7)")
        print(f"  'Python' vs 'weather' similarity: {sim_0_2:.3f} (should be low, <0.5)")
        
        # Test query vs document types
        print("\nTesting input types...")
        doc_emb = await client.embed_one("Python programming guide", input_type="document")
        query_emb = await client.embed_one("how to program in Python", input_type="query")
        
        sim = cosine_sim(doc_emb, query_emb)
        print(f"  Document-query similarity: {sim:.3f}")
        
        print("\n✅ Voyage embeddings test complete!")
    
    asyncio.run(test())
