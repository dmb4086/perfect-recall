#!/usr/bin/env python3
"""
Kimi (Moonshot) Embeddings Client

Uses Moonshot's OpenAI-compatible API for text embeddings.
"""

import os
import asyncio
import hashlib
from typing import List, Optional
import aiohttp


class KimiEmbeddings:
    """Client for Moonshot AI embeddings API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.moonshot.cn/v1",
        model: str = "moonshot-embedding-text-1",
        dimension: int = 1536,
    ):
        self.api_key = api_key or os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY") or os.getenv("KIMI_PLUGIN_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dimension = dimension
        
        if not self.api_key:
            raise ValueError(
                "Moonshot API key required. Set KIMI_API_KEY, MOONSHOT_API_KEY, or KIMI_PLUGIN_API_KEY environment variable."
            )
    
    async def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats (embedding vector)
        """
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
            payload = {
                "model": self.model,
                "input": text,
            }
            
            async with session.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json=payload,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Embedding API error: {response.status} - {error_text}")
                
                data = await response.json()
                return data["data"][0]["embedding"]
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        # Moonshot API supports batching in a single request
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
            payload = {
                "model": self.model,
                "input": texts,
            }
            
            async with session.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json=payload,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Embedding API error: {response.status} - {error_text}")
                
                data = await response.json()
                # Sort by index to maintain order
                embeddings = sorted(data["data"], key=lambda x: x["index"])
                return [e["embedding"] for e in embeddings]
    
    def get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]


class CachedKimiEmbeddings(KimiEmbeddings):
    """Kimi embeddings with local caching."""
    
    def __init__(self, cache_dir: str = ".embedding_cache", **kwargs):
        super().__init__(**kwargs)
        self.cache_dir = cache_dir
        self._cache: dict = {}
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_path(self, key: str) -> str:
        """Get path for cache entry."""
        return os.path.join(self.cache_dir, f"{key}.json")
    
    async def embed(self, text: str) -> List[float]:
        """Get embedding with caching."""
        cache_key = self.get_cache_key(text)
        
        # Check memory cache
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Check disk cache
        cache_path = self._get_cache_path(cache_key)
        if os.path.exists(cache_path):
            import json
            with open(cache_path) as f:
                embedding = json.load(f)
            self._cache[cache_key] = embedding
            return embedding
        
        # Fetch from API
        embedding = await super().embed(text)
        
        # Save to caches
        self._cache[cache_key] = embedding
        import json
        with open(cache_path, 'w') as f:
            json.dump(embedding, f)
        
        return embedding


# Convenience function for integration
def get_embedding_func(api_key: Optional[str] = None):
    """
    Get an embedding function compatible with memory system.
    
    Returns a sync function that wraps the async embed.
    """
    client = CachedKimiEmbeddings(api_key=api_key)
    
    def embed_sync(text: str) -> List[float]:
        """Synchronous wrapper for async embed."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, create task
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(client.embed(text))
            else:
                return loop.run_until_complete(client.embed(text))
        except RuntimeError:
            # No loop running, create new one
            return asyncio.run(client.embed(text))
    
    return embed_sync


# Test
if __name__ == "__main__":
    async def test():
        # Check for API key
        api_key = os.getenv("MOONSHOT_API_KEY") or os.getenv("KIMI_API_KEY")
        
        if not api_key:
            print("❌ No API key found. Set MOONSHOT_API_KEY or KIMI_API_KEY.")
            print("Running in mock mode...")
            
            # Mock embeddings for testing
            def mock_embed(text: str) -> list[float]:
                import numpy as np
                hash_bytes = hashlib.sha256(text.encode()).digest()
                np.random.seed(int.from_bytes(hash_bytes[:4], 'big'))
                embedding = np.random.randn(1536).astype(np.float32)
                embedding = embedding / np.linalg.norm(embedding)
                return embedding.tolist()
            
            # Test mock
            emb1 = mock_embed("hello world")
            emb2 = mock_embed("hello world")
            emb3 = mock_embed("goodbye world")
            
            print(f"Mock embedding dim: {len(emb1)}")
            print(f"Same text, same embedding: {emb1 == emb2}")
            print(f"Different text, different embedding: {emb1 != emb3}")
            return
        
        # Real API test
        client = CachedKimiEmbeddings(api_key=api_key)
        
        print("Testing Kimi embeddings...")
        
        # Single embed
        text = "Perfect Recall memory system"
        print(f"\nEmbedding: '{text}'")
        
        start = asyncio.get_event_loop().time()
        embedding = await client.embed(text)
        elapsed = asyncio.get_event_loop().time() - start
        
        print(f"  Dimension: {len(embedding)}")
        print(f"  First 5 values: {embedding[:5]}")
        print(f"  Latency: {elapsed:.3f}s")
        
        # Test caching
        print("\nTesting cache...")
        start = asyncio.get_event_loop().time()
        embedding2 = await client.embed(text)
        elapsed = asyncio.get_event_loop().time() - start
        print(f"  Cached lookup latency: {elapsed:.3f}s (should be ~0)")
        
        # Test semantic similarity
        print("\nTesting semantic similarity...")
        texts = [
            "I love Python programming",
            "Python is my favorite language",
            "The weather is nice today",
        ]
        
        embeddings = []
        for t in texts:
            emb = await client.embed(t)
            embeddings.append(emb)
        
        # Calculate cosine similarities
        import numpy as np
        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        sim_0_1 = cosine_sim(embeddings[0], embeddings[1])
        sim_0_2 = cosine_sim(embeddings[0], embeddings[2])
        
        print(f"  'Python' vs 'Python' similarity: {sim_0_1:.3f} (should be high)")
        print(f"  'Python' vs 'weather' similarity: {sim_0_2:.3f} (should be low)")
        
        print("\n✅ Kimi embeddings test complete!")
    
    asyncio.run(test())
