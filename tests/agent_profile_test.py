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
from controller.stance_analysis.agent_profile_store import AgentProfileStore

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

    # In-memory rolling store of posts (candidate items)
    analyzer = EmbeddingAnalyzer(topic)
    posts = RollingEmbeddingStore(topic=topic, analyzer=analyzer, redis_cache=None)

    seed_posts = [
        f"{topic} is obviously necessary. People fighting it are holding us back.",
        f"I’m not convinced about {topic}. There are real downsides nobody wants to discuss.",
        f"On {topic}, I see valid points on both sides. I want more evidence.",
        f"Stop pretending {topic} is harmless — it’s going to backfire.",
        f"{topic} is a win for everyday people. Support it loudly.",
    ]

    for i, text in enumerate(seed_posts):
        await posts.add(text, id=f"seed-{i}", metadata={"sender_id": "seed"})

    # Precomputed agent profile (sliding window)
    store = AgentProfileStore(redis=None, window_size=10, seed_weight=5.0)

    agent_id = "agent_1"
    init_prompt = f"You are a social media persona with strong opinions about {topic}."
    await store.ensure_initialized(agent_id, seed_text=init_prompt, topic_for_embedding=topic)

    # Simulate interactions
    await store.add_interaction(agent_id, text=f"{topic} is a disaster and we need to stop it.", interaction_type="consumed", topic=topic)
    await store.add_interaction(agent_id, text=f"I support {topic} because it helps ordinary people.", interaction_type="authored", topic=topic)
    await store.add_interaction(agent_id, text=f"On {topic}, I’m skeptical. The costs are too high.", interaction_type="consumed", topic=topic)

    agent_view = await store.get_agent_topic_view(agent_id, topic=topic)
    if not agent_view:
        raise RuntimeError("Failed to score agent")

    recs = await posts.recommend_for_agent_vector(
        agent_vector=(await store.load(agent_id)).vector,
        agent_stance_score=agent_view["stance_score"],
        agent_strength=agent_view["strength"],
        top_k=3,
    )

    print(f"Topic: {topic}")
    print(f"Agent stance_score: {agent_view['stance_score']:.3f} strength: {agent_view['strength']:.3f} topic_similarity: {agent_view['topic_similarity']:.3f}")
    print("\nTop recommendations for agent:")
    for r in recs:
        print(f"- d={r['distance']:.3f} stance={r['stance_score']:.3f} strength={r['strength']:.3f} :: {r['text']}")




if __name__ == "__main__":
    asyncio.run(main())
