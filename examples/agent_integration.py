"""
Agent Integration Example

Shows how to integrate Perfect Recall with an AI agent.
This is a mock implementation - replace with your actual LLM client.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from perfect_recall import PerfectRecall, EpisodeType


class MockLLMClient:
    """Mock LLM client for demonstration."""
    
    async def chat(self, messages: list[dict]) -> str:
        """Mock chat completion."""
        # Simple mock response
        user_msg = messages[-1].get("content", "")
        
        if "debug" in user_msg.lower():
            return "I see you're debugging something. Let me help you trace through the issue."
        elif "python" in user_msg.lower():
            return "Python is a great choice! I can help you with that."
        else:
            return "I understand. Tell me more about what you're working on."


class MemoryEnhancedAgent:
    """
    An AI agent enhanced with Perfect Recall memory.
    
    This agent:
    1. Retrieves relevant memories before responding
    2. Records interactions to memory
    3. Maintains working memory across turns
    """
    
    def __init__(self):
        self.pr: PerfectRecall = None
        self.llm = MockLLMClient()
        self.current_session = None
    
    async def initialize(self):
        """Initialize the agent and memory system."""
        self.pr = await PerfectRecall.create(
            database_url=os.getenv(
                "DATABASE_URL",
                "postgresql+asyncpg://perfect_recall:perfect_recall_secret@localhost:5432/perfect_recall"
            ),
            embedding_func=self._mock_embedding,
        )
    
    async def shutdown(self):
        """Clean up resources."""
        if self.pr:
            await self.pr.close()
    
    def _mock_embedding(self, text: str) -> list[float]:
        """Generate mock embedding."""
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        return [(hash_val % 1000) / 1000.0 for _ in range(1536)]
    
    async def start_conversation(self, user_id: str) -> str:
        """Start a new conversation session."""
        self.current_session = await self.pr.start_session(
            user_id=user_id,
            agent_id="memory_enhanced_agent",
        )
        return str(self.current_session.id)
    
    async def chat(self, message: str) -> str:
        """
        Process a user message and generate a response.
        
        This method:
        1. Retrieves relevant memories
        2. Builds context-augmented prompt
        3. Generates response
        4. Records interaction to memory
        """
        if not self.current_session:
            raise RuntimeError("No active session. Call start_conversation first.")
        
        # Step 1: Retrieve relevant memories
        memories = await self.pr.recall_for_session(
            session_id=str(self.current_session.id),
            query=message,
            include_working_memory=True,
            limit=5,
        )
        
        # Step 2: Build context-augmented prompt
        memory_context = self._format_memories(memories)
        
        prompt = f"""You are a helpful assistant with memory of past conversations.

{memory_context}

User: {message}
Assistant:"""
        
        # Step 3: Generate response (using mock LLM)
        response = await self.llm.chat([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ])
        
        # Step 4: Record interaction to memory
        await self.pr.record_episode(
            content=f"User: {message}\nAssistant: {response}",
            episode_type=EpisodeType.MESSAGE,
            session_id=self.current_session.id,
        )
        
        # Also store any explicit facts mentioned
        await self._extract_and_store_facts(message)
        
        return response
    
    def _format_memories(self, memories: list) -> str:
        """Format memories for the prompt."""
        if not memories:
            return "[No previous context]"
        
        lines = ["## Relevant Context from Memory:"]
        for mem in memories:
            lines.append(f"- {mem.memory.content}")
        
        return "\n".join(lines)
    
    async def _extract_and_store_facts(self, message: str):
        """Extract and store facts from user message."""
        # Simple rule-based extraction
        # In production, use an LLM for this
        
        message_lower = message.lower()
        
        # Extract preferences
        if "i prefer" in message_lower or "i like" in message_lower:
            # Store as semantic fact
            # (In production, parse more carefully)
            pass
    
    async def end_conversation(self):
        """End the current conversation."""
        if self.current_session:
            await self.pr.end_session(self.current_session.id)
            self.current_session = None


async def demo():
    """Run a demo conversation."""
    print("=" * 60)
    print("Memory-Enhanced Agent Demo")
    print("=" * 60)
    
    agent = MemoryEnhancedAgent()
    await agent.initialize()
    
    try:
        # Start conversation
        print("\n1. Starting conversation...")
        session_id = await agent.start_conversation("demo_user_123")
        print(f"   Session: {session_id}")
        
        # Simulate conversation turns
        conversation = [
            "Hi, I need help debugging a Python script",
            "I'm getting a KeyError on line 42",
            "By the way, I prefer using Python for data analysis",
            "Can you remind me what we were debugging?",
        ]
        
        for i, message in enumerate(conversation, 1):
            print(f"\n2.{i} Conversation turn:")
            print(f"   User: {message}")
            
            response = await agent.chat(message)
            print(f"   Agent: {response}")
        
        # End conversation
        print("\n3. Ending conversation...")
        await agent.end_conversation()
        print("   ✓ Conversation saved to memory")
        
    finally:
        await agent.shutdown()
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    if os.getenv("SKIP_DB"):
        print("SKIP_DB set - skipping database operations")
        sys.exit(0)
    
    try:
        asyncio.run(demo())
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure PostgreSQL is running with:")
        print("  docker-compose up -d")
        sys.exit(1)
