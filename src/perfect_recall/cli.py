#!/usr/bin/env python3
"""
Perfect Recall CLI - Simple command-line interface for testing.

Usage:
    python -m perfect_recall.cli init          # Initialize database
    python -m perfect_recall.cli store "text"  # Store a memory
    python -m perfect_recall.cli recall "query" # Recall memories
    python -m perfect_recall.cli interactive   # Interactive mode
"""

import asyncio
import argparse
import sys
from typing import Optional, List
from uuid import UUID

# Add src to path for imports
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from perfect_recall.core.perfect_recall import PerfectRecall
from perfect_recall.core.abstention import AbstentionController
from perfect_recall.models.memory import EpisodeType
from perfect_recall.db.connection import DatabaseManager


def get_database_url() -> str:
    """Get database URL from environment or default."""
    return os.environ.get(
        "DATABASE_URL",
        "postgresql://perfect_recall:perfect_recall_secret@localhost:5432/perfect_recall"
    )


def mock_embedding(text: str) -> List[float]:
    """
    Mock embedding function for testing without API keys.
    Generates deterministic embeddings based on text content.
    """
    import hashlib
    import numpy as np
    
    # Use hash to generate deterministic embedding
    hash_bytes = hashlib.sha256(text.encode()).digest()
    
    # Generate 1536-dimensional vector (OpenAI dimension)
    np.random.seed(int.from_bytes(hash_bytes[:4], 'big'))
    embedding = np.random.randn(1536).astype(np.float32)
    
    # Normalize to unit vector
    embedding = embedding / np.linalg.norm(embedding)
    
    return embedding.tolist()


class PerfectRecallCLI:
    """Command-line interface for Perfect Recall."""
    
    def __init__(self):
        self.pr: Optional[PerfectRecall] = None
        self.abstention = AbstentionController()
        self.current_session_id: Optional[str] = None
    
    async def init(self) -> bool:
        """Initialize the Perfect Recall system."""
        try:
            print("🔄 Initializing Perfect Recall...")
            self.pr = await PerfectRecall.create(
                database_url=get_database_url(),
                embedding_func=mock_embedding,
            )
            
            # Health check
            health = await self.pr.health_check()
            if health["status"] == "healthy":
                print("✅ Perfect Recall initialized successfully!")
                print(f"   Database: {get_database_url()}")
                return True
            else:
                print("❌ Health check failed")
                return False
                
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            print("\n💡 Make sure Postgres is running:")
            print("   docker-compose up -d")
            return False
    
    async def close(self):
        """Close the Perfect Recall system."""
        if self.pr:
            await self.pr.close()
    
    async def cmd_init(self, args):
        """Initialize database schema."""
        if await self.init():
            print("\n📊 Database schema initialized")
            print("\nNext steps:")
            print("  - Store memories: python -m perfect_recall.cli store 'Your memory'")
            print("  - Recall: python -m perfect_recall.cli recall 'Your query'")
            print("  - Interactive: python -m perfect_recall.cli interactive")
    
    async def cmd_store(self, args):
        """Store a memory."""
        if not await self.init():
            return
        
        content = " ".join(args.content)
        
        # Start session if needed
        if not self.current_session_id:
            session = await self.pr.start_session(user_id="cli_user")
            self.current_session_id = str(session.id)
            print(f"📍 Session started: {self.current_session_id[:8]}...")
        
        # Determine episode type
        episode_type = EpisodeType.MESSAGE
        if args.type:
            try:
                episode_type = EpisodeType(args.type.lower())
            except ValueError:
                print(f"⚠️  Unknown type '{args.type}', using 'message'")
        
        # Store the memory
        memory = await self.pr.record_episode(
            content=content,
            episode_type=episode_type,
            session_id=UUID(self.current_session_id) if self.current_session_id else None,
        )
        
        if memory:
            print(f"✅ Memory stored successfully!")
            print(f"   ID: {memory.id}")
            print(f"   Tier: {memory.memory_tier.value}")
            print(f"   Content: {content[:80]}...")
        else:
            print("🚫 Memory rejected by write gate (not significant enough)")
    
    async def cmd_recall(self, args):
        """Recall memories."""
        if not await self.init():
            return
        
        query = " ".join(args.query)
        limit = args.limit or 5
        
        print(f"🔍 Query: \"{query}\"")
        print(f"   Limit: {limit}")
        print()
        
        # Retrieve memories
        results = await self.pr.recall(
            query=query,
            session_id=self.current_session_id,
            limit=limit,
        )
        
        # Check abstention
        decision = self.abstention.should_abstain(query, results)
        
        if decision.abstain:
            print(self.abstention.get_abstention_message(decision, query))
            return
        
        # Display results
        if not results:
            print("📭 No memories found")
            return
        
        print(f"📚 Retrieved {len(results)} memories (confidence: {decision.confidence:.2f}):\n")
        
        for i, result in enumerate(results, 1):
            mem = result.memory
            print(f"{i}. [{mem.memory_tier.value.upper()}] (salience: {result.salience_score:.2f})")
            print(f"   Content: {mem.content[:100]}...")
            
            if mem.triggers:
                print(f"   Triggers: {', '.join(mem.triggers[:3])}")
            if mem.aliases:
                print(f"   Aliases: {', '.join(mem.aliases[:3])}")
            print()
    
    async def cmd_session(self, args):
        """Manage sessions."""
        if not await self.init():
            return
        
        if args.action == "new":
            session = await self.pr.start_session(user_id="cli_user")
            self.current_session_id = str(session.id)
            print(f"✅ New session started: {self.current_session_id}")
            
        elif args.action == "end":
            if self.current_session_id:
                await self.pr.end_session(UUID(self.current_session_id))
                print(f"✅ Session ended: {self.current_session_id[:8]}...")
                self.current_session_id = None
            else:
                print("⚠️  No active session")
                
        elif args.action == "status":
            if self.current_session_id:
                print(f"📍 Active session: {self.current_session_id}")
            else:
                print("📍 No active session")
    
    async def cmd_interactive(self, args):
        """Run interactive mode."""
        if not await self.init():
            return
        
        print("\n" + "="*60)
        print("🧠 Perfect Recall - Interactive Mode")
        print("="*60)
        print("\nCommands:")
        print("  store <text>  - Store a memory")
        print("  recall <text> - Recall memories")
        print("  session       - Show current session")
        print("  quit          - Exit")
        print()
        
        # Start a session
        session = await self.pr.start_session(user_id="cli_user")
        self.current_session_id = str(session.id)
        print(f"📍 Session: {self.current_session_id[:8]}...")
        print()
        
        while True:
            try:
                user_input = input("> ").strip()
                
                if not user_input:
                    continue
                
                parts = user_input.split()
                cmd = parts[0].lower()
                
                if cmd == "quit" or cmd == "exit":
                    print("👋 Goodbye!")
                    break
                    
                elif cmd == "store":
                    if len(parts) < 2:
                        print("⚠️  Usage: store <text>")
                        continue
                    
                    content = " ".join(parts[1:])
                    memory = await self.pr.record_episode(
                        content=content,
                        episode_type=EpisodeType.MESSAGE,
                        session_id=UUID(self.current_session_id),
                    )
                    
                    if memory:
                        print(f"✅ Stored: {content[:60]}...")
                    else:
                        print("🚫 Rejected by write gate")
                
                elif cmd == "recall":
                    if len(parts) < 2:
                        print("⚠️  Usage: recall <query>")
                        continue
                    
                    query = " ".join(parts[1:])
                    results = await self.pr.recall(
                        query=query,
                        session_id=self.current_session_id,
                        limit=5,
                    )
                    
                    # Check abstention
                    decision = self.abstention.should_abstain(query, results)
                    
                    if decision.abstain:
                        print(self.abstention.get_abstention_message(decision, query))
                        continue
                    
                    if results:
                        print(f"\n📚 {len(results)} results (confidence: {decision.confidence:.2f}):\n")
                        for i, result in enumerate(results, 1):
                            print(f"{i}. [{result.memory.memory_tier.value.upper()}] "
                                  f"({result.salience_score:.2f})")
                            print(f"   {result.memory.content[:80]}...")
                        print()
                    else:
                        print("📭 No memories found")
                
                elif cmd == "session":
                    print(f"📍 Session: {self.current_session_id}")
                
                elif cmd == "help":
                    print("\nCommands:")
                    print("  store <text>  - Store a memory")
                    print("  recall <text> - Recall memories")
                    print("  session       - Show current session")
                    print("  quit          - Exit")
                    print()
                
                else:
                    print(f"⚠️  Unknown command: {cmd}")
                    print("   Type 'help' for available commands")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Perfect Recall CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s init                          # Initialize database
  %(prog)s store "I prefer Python"       # Store a memory
  %(prog)s recall "What do I prefer?"    # Recall memories
  %(prog)s interactive                   # Interactive mode
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # init command
    init_parser = subparsers.add_parser("init", help="Initialize database")
    
    # store command
    store_parser = subparsers.add_parser("store", help="Store a memory")
    store_parser.add_argument("content", nargs="+", help="Memory content")
    store_parser.add_argument("--type", "-t", help="Episode type (message, action, decision, etc.)")
    
    # recall command
    recall_parser = subparsers.add_parser("recall", help="Recall memories")
    recall_parser.add_argument("query", nargs="+", help="Query text")
    recall_parser.add_argument("--limit", "-l", type=int, default=5, help="Maximum results")
    
    # session command
    session_parser = subparsers.add_parser("session", help="Manage sessions")
    session_parser.add_argument("action", choices=["new", "end", "status"], 
                                help="Session action")
    
    # interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Interactive mode")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    cli = PerfectRecallCLI()
    
    try:
        if args.command == "init":
            asyncio.run(cli.cmd_init(args))
        elif args.command == "store":
            asyncio.run(cli.cmd_store(args))
        elif args.command == "recall":
            asyncio.run(cli.cmd_recall(args))
        elif args.command == "session":
            asyncio.run(cli.cmd_session(args))
        elif args.command == "interactive":
            asyncio.run(cli.cmd_interactive(args))
    except KeyboardInterrupt:
        print("\n👋 Interrupted")
    finally:
        if cli.pr:
            asyncio.run(cli.close())


if __name__ == "__main__":
    main()
