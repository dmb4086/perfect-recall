import asyncio
import json
import sys
import os
from pathlib import Path

# Add src to python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../tests'))

from perfect_recall.core.perfect_recall import PerfectRecall
from perfect_recall.core.bm25_retriever import BM25Retriever
from generate_synthetic_memory import SYNTHETIC_PAIRS, generate_memory_md, generate_queries_json
from unittest.mock import AsyncMock, MagicMock

class MockMemory:
    def __init__(self, content):
        self.content = content

class MockRetrievedMemory:
    def __init__(self, content, score):
        self.memory = MockMemory(content)
        self.salience_score = score

class ShadowValidator:
    def __init__(self):
        self.pr = None
        self.bm25 = None
        self.queries = []
        self.pr_mock_db = []
        self.using_mock_pr = False

    def _mock_embedding(self, text: str) -> list[float]:
        """Generate deterministic pseudo-embeddings for fallback-only smoke testing."""
        import hashlib
        import numpy as np

        # Hash-seeded vectors are deterministic but not semantically meaningful.
        hash_bytes = hashlib.sha256(text.encode()).digest()
        np.random.seed(int.from_bytes(hash_bytes[:4], 'big'))
        embedding = np.random.randn(1536).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()

    async def _mock_recall(self, query: str, limit: int = 5, **kwargs):
        """A simple cosine-similarity recall method to mock PR when DB is unavailable"""
        import numpy as np
        query_emb = np.array(self._mock_embedding(query))

        results = []
        for mem in self.pr_mock_db:
            mem_emb = np.array(mem["embedding"])
            # Cosine similarity
            score = np.dot(query_emb, mem_emb)
            results.append((mem["content"], score))

        # Sort descending by score
        results.sort(key=lambda x: x[1], reverse=True)

        # Return top K as mock objects
        return [MockRetrievedMemory(content, score) for content, score in results[:limit]]

    async def initialize(self):
        # 1. Generate the synthetic data
        repo_root = Path(__file__).parent.parent
        memory_md_path = repo_root / "synthetic_memory.md"
        queries_json_path = repo_root / "synthetic_queries.json"

        if not memory_md_path.exists() or not queries_json_path.exists():
            print("Generating synthetic dataset...")
            generate_memory_md(SYNTHETIC_PAIRS, str(memory_md_path))
            generate_queries_json(SYNTHETIC_PAIRS, str(queries_json_path))
        else:
            print("Synthetic dataset already exists. Skipping generation.")

        with open(queries_json_path, "r") as f:
            self.queries = json.load(f)

        # 2. Initialize BM25 baseline
        print("Initializing BM25 Retriever...")
        self.bm25 = BM25Retriever(str(memory_md_path))

        # 3. Initialize Perfect Recall
        print("Initializing Perfect Recall...")
        try:
            # First try the real DB connection if docker is up
            self.pr = await PerfectRecall.create(
                database_url="postgresql+asyncpg://perfect_recall:perfect_recall_secret@localhost:5432/perfect_recall",
                embedding_func=self._mock_embedding,
            )

            # Clear existing memory for clean test
            async with self.pr.db.session() as session:
                from sqlalchemy import text
                await session.execute(text("TRUNCATE TABLE memory_nodes CASCADE;"))

            print("Ingesting data into Perfect Recall...")
            session_obj = await self.pr.start_session(user_id="benchmark_user")

            for pair in SYNTHETIC_PAIRS:
                await self.pr.store_fact(
                    subject="benchmark",
                    predicate="contains",
                    object=pair["memory"],
                    confidence=1.0
                )
        except Exception as e:
            print(f"Failed to connect to DB: {e}. Using mocked PR retriever.")
            print(
                "WARNING: Mock PR mode uses deterministic pseudo-embeddings and is "
                "not semantically comparable to real retrieval quality."
            )
            self.pr = MagicMock()
            self.pr.recall = AsyncMock(side_effect=self._mock_recall)
            self.pr.close = AsyncMock()
            self.using_mock_pr = True

            print("Ingesting data into Mock Perfect Recall...")
            for pair in SYNTHETIC_PAIRS:
                self.pr_mock_db.append({
                    "content": pair["memory"],
                    "embedding": self._mock_embedding(pair["memory"])
                })


    async def run_benchmark(self, k=3):
        print(f"\n--- Running Benchmark (Hit@{k}) ---\n")

        pr_hits = 0
        bm25_hits = 0
        total = len(self.queries)

        for item in self.queries:
            query = item["query"]
            expected = item["expected_memory"]

            # BM25 Retrieval
            bm25_results = self.bm25.search(query, limit=k)
            bm25_found = any(expected in res["content"] for res in bm25_results)
            if bm25_found:
                bm25_hits += 1

            # PR Retrieval
            pr_results = await self.pr.recall(query, limit=k)
            pr_found = any(expected in res.memory.content for res in pr_results)
            if pr_found:
                pr_hits += 1

        # Report
        print(f"Total Queries: {total}")
        print(f"BM25 Hit@{k}: {bm25_hits}/{total} ({(bm25_hits/total)*100:.1f}%)")

        if self.using_mock_pr:
            print(
                "PR Hit metrics were computed in MOCK mode and should not be used "
                "for quality comparisons."
            )
        print(f"PR Hit@{k}:   {pr_hits}/{total} ({(pr_hits/total)*100:.1f}%)")

        return {
            "total": total,
            "bm25_hits": bm25_hits,
            "pr_hits": pr_hits,
            "using_mock_pr": self.using_mock_pr,
        }

if __name__ == "__main__":
    async def main():
        validator = ShadowValidator()
        await validator.initialize()
        await validator.run_benchmark(k=3)
        await validator.pr.close()

    asyncio.run(main())
