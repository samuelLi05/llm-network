import os
import asyncio
import json
import random
import sys
from collections import Counter

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


def build_vaccines_posts() -> list[dict]:
    """Larger, semi-realistic vaccines post dataset for profile->recs testing."""
    posts: list[dict] = []

    def add(sender: str, label: str, text: str):
        posts.append({"sender_id": sender, "label": label, "text": text})

    # Mix of pro/anti/neutral, plus multiple subtopics (safety, mandates, trust)
    add("nurse_rn", "pro", "Vaccines prevent severe disease. I’m tired of watching misinformation win." )
    add("immunology_nerd", "pro", "mRNA doesn’t alter your DNA. Please stop repeating that myth." )
    add("public_health", "pro", "Higher coverage protects immunocompromised people. This is community care." )
    add("grandparent", "pro", "I remember polio. You do not want to relive that era." )
    add("pharmacist", "pro", "Side effects exist, but serious ones are rare and tracked. The disease is worse." )
    add("teacher_union", "pro", "Outbreaks disrupt schools and families. Vaccination helps keep life stable." )

    add("liberty_first", "anti", "Mandates cross a line. Consent matters even when you disagree." )
    add("skeptic_dad", "anti", "Pharma has conflicts of interest. Blind trust is not science." )
    add("injury_story", "anti", "I had a bad reaction and got dismissed. Don’t erase that experience." )
    add("alt_health", "anti", "We rushed policies and punished dissent. That’s not how you build trust." )
    add("risk_averse", "anti", "One-size-fits-all guidance ignores individual risk profiles." )
    add("online_skeptic", "anti", "Censoring questions made everything feel sketchier." )

    add("policy_wonk", "neutral", "Vaccines can work and mandates can still be ethically messy. Separate the questions." )
    add("curious_reader", "neutral", "I want clear risk/benefit by age group without the culture war." )
    add("family_doc", "neutral", "Talk to your doctor. Most people do fine, and context matters." )
    add("stats_person", "neutral", "What do hospitalization rates look like by cohort? That’s the real signal." )
    add("moderate_take", "neutral", "Shaming hesitant people is counterproductive. Explain, don’t attack." )
    add("civil_liberties", "neutral", "Public health and liberty conflict sometimes. We need a principled framework." )

    # Extra volume: paraphrases to create clusters
    add("immunology_nerd", "pro", "Vaccines train immunity; they don’t make you ‘dependent’." )
    add("peds_nurse", "pro", "These diseases can be brutal for kids. Prevention isn’t optional." )
    add("data_journalist", "pro", "Outcomes improved dramatically post-vaccine. Pretending otherwise is denial." )
    add("mandate_resister", "anti", "Employment mandates were coercion dressed up as ‘health’." )
    add("freedom_mom", "anti", "If it’s your choice, stop trying to force it. That contradiction matters." )
    add("policy_wonk", "neutral", "Targeted requirements in high-risk settings might be defensible; blanket rules less so." )

    return posts


def summarize(recs: list[dict]) -> dict:
    if not recs:
        return {"n": 0}
    stance_mean = sum(float(r.get("stance_score", 0.0)) for r in recs) / len(recs)
    labels = [((r.get("metadata") or {}).get("label") or "?") for r in recs]
    return {"n": len(recs), "stance_mean": stance_mean, "labels": Counter(labels)}


async def main():
    topics = load_topics(TOPICS_PATH)
    forced = (os.getenv("TEST_TOPIC") or "").strip().lower()
    topic = forced if forced else random.choice(topics)
    if topic == "vaccine":
        topic = "vaccines"

    # In-memory rolling store of posts (candidate items)
    analyzer = EmbeddingAnalyzer(topic)
    posts = RollingEmbeddingStore(topic=topic, analyzer=analyzer, redis_cache=None)

    # Build a richer corpus for better recommendation signal
    if topic == "vaccines":
        corpus = build_vaccines_posts()
    else:
        corpus = []
        for i in range(30):
            bucket = random.choice(["pro", "anti", "neutral"])
            sender = random.choice(["user_a", "user_b", "user_c", "user_d"])
            if bucket == "pro":
                text = f"I support {topic}. It’s the pragmatic path forward."
            elif bucket == "anti":
                text = f"I oppose {topic}. The risks are being minimized."
            else:
                text = f"On {topic}, I’m undecided. I want more evidence and tradeoffs."
            corpus.append({"sender_id": sender, "label": bucket, "text": text})

    for i, item in enumerate(corpus):
        await posts.add(
            item["text"],
            id=f"post-{i}",
            metadata={"sender_id": item.get("sender_id"), "label": item.get("label")},
        )

    # Precomputed agent profiles (sliding window)
    store = AgentProfileStore(redis=None, window_size=12, seed_weight=5.0)

    # Multiple personas: pro, anti, and neutral, plus a profile-drift check.
    personas = [
        (
            "agent_pro",
            f"You are a pro-vaccine advocate. You believe vaccines are safe and crucial for public health.",
            [
                "Vaccines save lives and misinformation is harming people.",
                "High vaccination rates protect the vulnerable.",
                "mRNA myths are embarrassing and dangerous.",
            ],
        ),
        (
            "agent_anti",
            f"You are strongly against vaccine mandates and skeptical of pharma incentives.",
            [
                "Mandates are coercive and undermine consent.",
                "Pharma conflicts of interest are real.",
                "People with side effects deserve to be heard.",
            ],
        ),
        (
            "agent_neutral",
            f"You are cautious and evidence-seeking about vaccines; you dislike polarization.",
            [
                "I want data by age group and clear risk/benefit.",
                "We should separate vaccine efficacy from mandate ethics.",
                "Shaming people doesn’t build trust.",
            ],
        ),
    ]

    print(f"Topic: {topic}")
    print(f"Corpus size: {len(corpus)}")

    if not bool(os.getenv("OPENAI_API_KEY")):
        print("\nWARNING: OPENAI_API_KEY not set; embeddings may fail at runtime.")

    for agent_id, init_prompt, interaction_texts in personas:
        await store.ensure_initialized(agent_id, seed_text=init_prompt, topic_for_embedding=topic)

        # Add enough interactions to test the sliding window cap
        for t in interaction_texts:
            await store.add_interaction(agent_id, text=t, interaction_type="consumed", topic=topic)
        for t in interaction_texts:
            await store.add_interaction(agent_id, text=f"My take: {t}", interaction_type="authored", topic=topic)

        profile = await store.load(agent_id)
        assert profile is not None and profile.vector is not None
        assert len(profile.window) <= profile.window_size

        agent_view = await store.get_agent_topic_view(agent_id, topic=topic)
        if not agent_view:
            raise RuntimeError("Failed to score agent")

        recs = await posts.recommend_for_agent_vector(
            agent_vector=profile.vector,
            agent_stance_score=agent_view["stance_score"],
            agent_strength=agent_view["strength"],
            top_k=10,
        )

        s = summarize(recs)
        print("\n" + "=" * 80)
        print(
            f"{agent_id} view: stance_score={agent_view['stance_score']:.3f} "
            f"strength={agent_view['strength']:.3f} topic_similarity={agent_view['topic_similarity']:.3f}"
        )
        print(f"Top-10 rec summary: stance_mean={s.get('stance_mean', 0.0):.3f} labels={dict(s.get('labels', {}))}")
        print("\nTop recommendations:")
        for r in recs:
            meta = r.get("metadata") or {}
            print(
                f"- d={r['distance']:.3f} stance={r['stance_score']:.3f} "
                f"label={meta.get('label','?')} sender={meta.get('sender_id','?')} :: {r['text']}"
            )

        # Weak directional expectations for vaccines (should usually hold)
        if topic == "vaccines":
            if agent_id == "agent_pro":
                assert float(s.get("stance_mean", 0.0)) > -0.2
            if agent_id == "agent_anti":
                assert float(s.get("stance_mean", 0.0)) < 0.2

    # Profile drift check: start neutral, then add strong anti interactions and confirm stance shifts
    drift_id = "agent_drift"
    await store.ensure_initialized(
        drift_id,
        seed_text=f"You are cautious and undecided about {topic}. You value evidence.",
        topic_for_embedding=topic,
    )
    before = await store.get_agent_topic_view(drift_id, topic=topic)
    if not before:
        raise RuntimeError("Failed to score drift agent (before)")

    for _ in range(6):
        await store.add_interaction(
            drift_id,
            text="Mandates are coercion and pharma incentives distort the truth.",
            interaction_type="consumed",
            topic=topic,
        )

    after = await store.get_agent_topic_view(drift_id, topic=topic)
    if not after:
        raise RuntimeError("Failed to score drift agent (after)")

    print("\n" + "=" * 80)
    print(
        f"Drift check: stance before={before['stance_score']:.3f} after={after['stance_score']:.3f} "
        f"(delta={after['stance_score']-before['stance_score']:+.3f})"
    )
    # Directional check: should move somewhat negative for vaccines axis
    if topic == "vaccines":
        assert after["stance_score"] <= before["stance_score"] + 0.05




if __name__ == "__main__":
    asyncio.run(main())
