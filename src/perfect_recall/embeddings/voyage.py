#!/usr/bin/env python3
"""
Voyage AI embedding client for Perfect Recall.
Best-of-best embeddings with fallback handling.
"""

import os
import asyncio
import time
import json
import hashlib
from typing import List, Optional
from dataclasses import dataclass

import httpx


@dataclass
class EmbeddingResult:
    embedding: List[float]
    model: str
    latency_ms: float
    cached: bool = False


class VoyageEmbedder:
    """
    Voyage AI embedder with:
    - Batched requests
    - Redis caching
    - Latency tracking
    """
    
    VOYAGE_API_URL = "https://api.voyageai.com/v1/embeddings"
    DEFAULT_MODEL = "voyage-3"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        redis_client = None,
    ):
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY")
        if not self.api_key:
            raise ValueError("Voyage API key required")
        
        self.model = model
        self.redis = redis_client
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def embed(
        self,
        texts: List[str],
        use_cache: bool = True,
    ) -> List[EmbeddingResult]:
        """
        Embed texts with caching.
        """
        if not texts:
            return []
        
        # Check cache first
        if use_cache and self.redis:
            cached_results = await self._get_cached(texts)
            # Filter out cached items
            texts_to_embed = [
                t for i, t in enumerate(texts) 
                if cached_results[i] is None
            ]
        else:
            cached_results = [None] * len(texts)
            texts_to_embed = texts
        
        # Embed uncached texts
        if texts_to_embed:
            api_results = await self._embed_batch(texts_to_embed)
            
            # Store in cache
            if use_cache and self.redis:
                await self._cache_results(texts_to_embed, api_results)
        else:
            api_results = []
        
        # Merge cached and fresh results
        results = []
        api_idx = 0
        for i, cached in enumerate(cached_results):
            if cached:
                results.append(cached)
            else:
                results.append(api_results[api_idx])
                api_idx += 1
        
        return results
    
    async def _embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """
        Call Voyage API for embeddings.
        """
        start = time.time()
        
        response = await self.client.post(
            self.VOYAGE_API_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "input": texts,
                "model": self.model,
            },
        )
        response.raise_for_status()
        
        latency_ms = (time.time() - start) * 1000
        data = response.json()
        
        results = []
        for item in data["data"]:
            results.append(EmbeddingResult(
                embedding=item["embedding"],
                model=self.model,
                latency_ms=latency_ms / len(texts),  # Per-item latency estimate
            ))
        
        return results
    
    async def _get_cached(self, texts: List[str]) -> List[Optional[EmbeddingResult]]:
        """Get cached embeddings from Redis."""
        if not self.redis:
            return [None] * len(texts)

        cached_results = []
        for text in texts:
            key = f"emb:{self.model}:{hashlib.md5(text.encode()).hexdigest()}"
            try:
                cached = await self.redis.get(key)
                if cached:
                    data = json.loads(cached)
                    cached_results.append(EmbeddingResult(
                        embedding=data["embedding"],
                        model=data["model"],
                        latency_ms=0.0,
                        cached=True
                    ))
                else:
                    cached_results.append(None)
            except Exception:
                # Fallback to no cache if redis fails
                cached_results.append(None)

        return cached_results
    
    async def _cache_results(
        self, 
        texts: List[str], 
        results: List[EmbeddingResult]
    ):
        """Cache embeddings in Redis."""
        if not self.redis:
            return

        for text, result in zip(texts, results):
            key = f"emb:{self.model}:{hashlib.md5(text.encode()).hexdigest()}"
            data = {
                "embedding": result.embedding,
                "model": result.model,
            }
            try:
                # Cache for 30 days
                await self.redis.setex(key, 30 * 24 * 60 * 60, json.dumps(data))
            except Exception:
                pass
    
    async def close(self):
        await self.client.aclose()


async def test_voyage():
    """Test Voyage API connectivity and latency."""
    embedder = VoyageEmbedder()
    
    test_texts = [
        "The quick brown fox jumps over the lazy dog",
        "Perfect Recall is a memory system for AI agents",
        "dev is building this at 4 AM",
    ]
    
    print(f"Testing Voyage API with model: {embedder.model}")
    print(f"Texts to embed: {len(test_texts)}")
    print("-" * 50)
    
    results = await embedder.embed(test_texts)
    
    for i, (text, result) in enumerate(zip(test_texts, results)):
        print(f"\nText {i+1}: {text[:50]}...")
        print(f"  Model: {result.model}")
        print(f"  Latency: {result.latency_ms:.2f}ms")
        print(f"  Dimensions: {len(result.embedding)}")
        print(f"  Sample values: {result.embedding[:3]}")
    
    await embedder.close()
    print("\n✅ Voyage API test complete")


if __name__ == "__main__":
    asyncio.run(test_voyage())
