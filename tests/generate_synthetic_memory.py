import json
import random
from datetime import datetime, timedelta

SYNTHETIC_PAIRS = [
    {
        "memory": "User prefers 2-space indentation for TypeScript projects.",
        "queries": ["what indentation style", "typescript formatting preference"],
        "category": "preferences"
    },
    {
        "memory": "Redis caching implemented for embedding API to reduce costs.",
        "queries": ["how to reduce API costs", "redis caching details", "why did we add redis"],
        "category": "architecture"
    },
    {
        "memory": "Team standup scheduled for 10 AM GMT+8 on weekdays.",
        "queries": ["when is the meeting", "standup time", "what time does the team meet"],
        "category": "schedule"
    },
    {
        "memory": "Database performance issues last week were caused by missing index on user_id column.",
        "queries": ["database performance", "what caused the outage", "missing index bug"],
        "category": "incidents"
    },
    {
        "memory": "Frontend uses Next.js with App Router and TailwindCSS for styling.",
        "queries": ["frontend stack", "react framework used", "CSS strategy"],
        "category": "architecture"
    },
    {
        "memory": "Always use absolute imports starting with '@/' instead of relative paths in React components.",
        "queries": ["import path conventions", "how to import in react"],
        "category": "conventions"
    },
    {
        "memory": "API rate limits are set to 100 requests per minute per IP address.",
        "queries": ["rate limiting config", "how many requests allowed"],
        "category": "configuration"
    },
    {
        "memory": "Authentication is handled by Supabase, specifically using magic links.",
        "queries": ["how does auth work", "supabase login"],
        "category": "architecture"
    },
    {
        "memory": "User typically takes lunch around 1 PM EST and is slow to respond then.",
        "queries": ["when is user at lunch", "slow response time"],
        "category": "preferences"
    },
    {
        "memory": "Production environment is hosted on Vercel, staging is on Heroku.",
        "queries": ["where is it deployed", "hosting provider"],
        "category": "infrastructure"
    },
    {
        "memory": "Stripe webhooks are processed in the 'api/webhooks/stripe' endpoint.",
        "queries": ["payment processing", "where do webhooks go"],
        "category": "architecture"
    },
    {
        "memory": "Background jobs are handled using Celery and Redis.",
        "queries": ["how are async tasks run", "celery setup"],
        "category": "architecture"
    },
    {
        "memory": "User prefers concise, bulleted responses without conversational filler.",
        "queries": ["how should I format my responses", "communication style"],
        "category": "preferences"
    },
    {
        "memory": "PostgreSQL is used as the primary relational database.",
        "queries": ["what db are we using", "relational storage"],
        "category": "architecture"
    },
    {
        "memory": "Use snake_case for Python variables and camelCase for JavaScript variables.",
        "queries": ["naming conventions", "how to name variables"],
        "category": "conventions"
    },
    {
        "memory": "The 'admin' role has full access, while 'editor' can only modify content but not settings.",
        "queries": ["user roles and permissions", "what can an editor do"],
        "category": "authorization"
    },
    {
        "memory": "Deployments happen automatically when a PR is merged into the 'main' branch.",
        "queries": ["how to deploy", "CI/CD process"],
        "category": "infrastructure"
    },
    {
        "memory": "Log files are rotated daily and kept for 30 days before deletion.",
        "queries": ["log retention policy", "how long do we keep logs"],
        "category": "configuration"
    },
    {
        "memory": "The user hates when the AI suggests using jQuery.",
        "queries": ["technologies to avoid", "jquery opinion"],
        "category": "preferences"
    },
    {
        "memory": "Feature flags are managed using LaunchDarkly.",
        "queries": ["how do we turn features on and off", "feature toggle tool"],
        "category": "architecture"
    }
]

def generate_memory_md(pairs, output_path="synthetic_memory.md"):
    """Generates a realistic-looking MEMORY.md file."""
    base_time = datetime.now() - timedelta(days=30)

    with open(output_path, "w") as f:
        f.write("# Perfect Recall Synthetic Ground Truth\n\n")
        f.write("This file contains synthetic memories used for benchmarking retrieval systems.\n\n")

        for i, pair in enumerate(pairs):
            # Generate a realistic-looking timestamp
            timestamp = base_time + timedelta(days=random.randint(0, 30), hours=random.randint(0, 23))
            timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")

            f.write(f"## [{timestamp_str}] {pair['category'].capitalize()} Note\n")
            f.write(f"Tags: #{pair['category']}\n\n")
            f.write(f"{pair['memory']}\n\n")

def generate_queries_json(pairs, output_path="synthetic_queries.json"):
    """Generates a JSON file mapping queries to their expected memory strings."""
    benchmark_data = []

    for pair in pairs:
        for query in pair["queries"]:
            benchmark_data.append({
                "query": query,
                "expected_memory": pair["memory"],
                "category": pair["category"]
            })

    with open(output_path, "w") as f:
        json.dump(benchmark_data, f, indent=2)

if __name__ == "__main__":
    generate_memory_md(SYNTHETIC_PAIRS)
    generate_queries_json(SYNTHETIC_PAIRS)
    print(f"Generated synthetic_memory.md with {len(SYNTHETIC_PAIRS)} entries.")
    print(f"Generated synthetic_queries.json with {sum(len(p['queries']) for p in SYNTHETIC_PAIRS)} queries.")
