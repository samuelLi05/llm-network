import os
import asyncio
import json
import random
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from controller.stance_analysis.embedding_analyzer import EmbeddingAnalyzer
from controller.stance_analysis.rolling_embedding_store import RollingEmbeddingStore

TOPICS_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "agents",
    "prompt_configs",
    "random_prompt.json",
)


def load_topics(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("topics", [])


async def main():
    topics = load_topics(TOPICS_PATH)
    topic = random.choice(topics)

    analyzer = EmbeddingAnalyzer(topic)
    store = RollingEmbeddingStore(topic=topic, analyzer=analyzer, redis_cache=None)

    seed_posts = [
        f"{topic} is obviously necessary. People fighting it are holding us back.",
        f"I’m not convinced about {topic}. There are real downsides nobody wants to discuss.",
        f"On {topic}, I see valid points on both sides. I want more evidence.",
        f"Stop pretending {topic} is harmless — it’s going to backfire.",
        f"{topic} is a win for everyday people. Support it loudly.",
    ]

    for i, text in enumerate(seed_posts):
        await store.add(text, id=f"seed-{i}", metadata={"sender_id": "seed"})

    query = f"Honestly, {topic} has tradeoffs but I’m leaning supportive if it’s implemented carefully."
    recs = await store.recommend(query, top_k=3)

    print(f"Topic: {topic}")
    print(f"Query: {query}")
    print("\nTop recommendations:")
    for r in recs:
        print(f"- d={r['distance']:.3f} stance={r['stance_score']:.3f} strength={r['strength']:.3f} :: {r['text']}")


if __name__ == "__main__":
    asyncio.run(main())
