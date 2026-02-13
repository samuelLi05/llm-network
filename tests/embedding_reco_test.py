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


def build_vaccines_corpus() -> list[dict]:
    """A larger, semi-realistic vaccines corpus with stance labels.

    Labels are only for analysis/printing in this test; they are not used by the model.
    """
    topic = "vaccines"
    posts: list[dict] = []

    def add(sender: str, label: str, text: str):
        posts.append(
            {
                "topic": topic,
                "sender_id": sender,
                "label": label,
                "text": text,
            }
        )

    # Pro / strongly supportive
    add("nurse_rn", "pro", "Vaccines are one of the most effective public health tools we have. You don’t get to pretend diseases vanish on their own.")
    add("epi_grad", "pro", "If you claim vaccines ‘don’t work’, explain why outbreaks explode when coverage drops. The data is not subtle.")
    add("dad_of_two", "pro", "Got my kids vaccinated. Zero drama, peace of mind. The fear-mongering is exhausting.")
    add("immunology_nerd", "pro", "No, mRNA vaccines don’t ‘rewrite your DNA’. That’s not how any of this works.")
    add("public_health", "pro", "Herd immunity isn’t a vibe—it’s math. High coverage protects people who can’t be vaccinated.")
    add("oncology_patient", "pro", "My immune system is weak. Your vaccine choice affects people like me. Please be normal about this.")
    add("teacher_union", "pro", "School outbreaks derail learning. Vaccination keeps classrooms open and kids safe.")
    add("pharmacist", "pro", "Adverse events are monitored constantly. ‘Nobody talks about side effects’ is just false.")
    add("data_journalist", "pro", "The conspiracy narrative collapses the moment you look at actual hospitalization and death rates.")
    add("mom_science", "pro", "I used to be scared too—then I read primary sources. Vaccines aren’t perfect, but they’re far safer than the diseases.")

    # Anti / strongly skeptical
    add("liberty_first", "anti", "I don’t care how you spin it: medical mandates are a hard no. My body, my choice actually has to mean something.")
    add("skeptic_dad", "anti", "Pharma companies have lied before. Acting like they’re saints now is naive.")
    add("alt_health", "anti", "We’re injecting kids with products that got rushed approvals and we’re supposed to clap? Pass.")
    add("injury_story", "anti", "I had a bad reaction after a vaccine and got gaslit. Don’t tell me it ‘never happens’.")
    add("wellness_guru", "anti", "Natural immunity is real. We should focus on nutrition and prevention instead of endless shots.")
    add("freedom_mom", "anti", "If the product is ‘safe and effective’, why are people so desperate to force it on everyone?")
    add("mandate_resister", "anti", "Employment mandates turned neighbors into snitches. That damage doesn’t disappear.")
    add("concerned_citizen", "anti", "We need real transparency on long-term effects, not slogans.")
    add("anti_authority", "anti", "Trust is earned. Public health lost credibility when dissent got censored.")
    add("risk_averse", "anti", "Maybe vaccines help some people, but one-size-fits-all policy is reckless.")

    # Neutral / mixed / cautious
    add("curious_reader", "neutral", "I’m pro-science but I also want clear explanations of risks vs benefits. Can we stop with the tribal shouting?")
    add("policy_wonk", "neutral", "Vaccines can reduce severe outcomes, but mandates raise ethical questions. Both can be true.")
    add("family_doc", "neutral", "Most patients do fine, some have side effects. Talk to your doctor instead of TikTok.")
    add("stats_person", "neutral", "What’s the baseline risk by age group? Without that, the conversation is just vibes.")
    add("moderate_take", "neutral", "I’m okay with vaccines, not okay with treating hesitant people like villains. Persuasion beats punishment.")
    add("civil_liberties", "neutral", "Public health and civil liberties are always in tension. We need a framework, not hysteria.")
    add("realist", "neutral", "Some people are scared because institutions messed up before. You can’t shame people into trust.")
    add("open_minded", "neutral", "I want more data on boosters: who benefits most and when?" )
    add("health_econ", "neutral", "The cost of outbreaks is real, but so is the cost of coercive policy. Evaluate both honestly." )
    add("skeptical_but_ok", "neutral", "I got vaccinated, but I still think we should be able to question guidance without being labeled anti-science." )

    # Add more "social network" noise/posts (multiple subtopics)
    add("mom_group", "neutral", "I’m trying to decide about flu shots for my kids—what are the most reputable sources?" )
    add("bio_prof", "pro", "If you’re worried about ingredients, learn what adjuvants actually do. It’s not a mystery potion." )
    add("freedom_rally", "anti", "Mandates were a stress test and we failed. Never again." )
    add("peds_nurse", "pro", "I’ve seen infants hospitalized. These diseases are not ‘just a cold’." )
    add("local_news", "neutral", "Our county’s vaccination rate fell this year. Officials warn of increased outbreak risk." )
    add("community_org", "pro", "Access matters: mobile clinics and paid time off help more than yelling at people online." )
    add("online_skeptic", "anti", "Every time questions get banned, it makes me less likely to believe the official story." )
    add("grandparent", "pro", "I remember polio. People arguing against vaccines have no idea what they’re inviting back." )
    add("mandate_debate", "neutral", "Could we target mandates to high-risk settings instead of blanket rules?" )
    add("diet_push", "anti", "Instead of shots, fix metabolic health. That’s the real root cause." )

    # Duplicate-ish themes with different phrasing (to test clustering)
    add("immunology_nerd", "pro", "Vaccines train your immune system. They don’t make you ‘dependent’." )
    add("injury_story", "anti", "People who had side effects deserve support, not ridicule." )
    add("policy_wonk", "neutral", "We should separate ‘vaccines work’ from ‘mandates are justified’." )
    add("dad_of_two", "pro", "My kid’s pediatrician answered every question. That’s what convinced me." )
    add("risk_averse", "anti", "If someone has prior reactions, forcing the same protocol is irresponsible." )

    return posts


def build_generic_corpus(topic: str, *, n: int = 30) -> list[dict]:
    """Fallback corpus for random topics when not forcing vaccines."""
    senders = ["user_a", "user_b", "user_c", "user_d", "user_e", "user_f"]
    pro_templates = [
        "{topic} is overdue. The benefits are obvious if you care about progress.",
        "Backing {topic} is just practical. Let’s stop stalling.",
        "I’m strongly in favor of {topic}. People fear change more than harm.",
    ]
    anti_templates = [
        "{topic} is a mistake. The risks are being waved away.",
        "I oppose {topic}. It’s costly and the downsides will hit regular people.",
        "We should push back on {topic}. This isn’t ‘progress’, it’s chaos.",
    ]
    neutral_templates = [
        "I’m torn on {topic}. I want more evidence before taking a side.",
        "{topic} has tradeoffs. The best approach is probably somewhere in the middle.",
        "Not convinced either way on {topic}. Show me data, not slogans.",
    ]
    posts: list[dict] = []
    for i in range(n):
        sender = random.choice(senders)
        bucket = random.choice(["pro", "anti", "neutral"])
        template = random.choice({"pro": pro_templates, "anti": anti_templates, "neutral": neutral_templates}[bucket])
        posts.append(
            {
                "topic": topic,
                "sender_id": sender,
                "label": bucket,
                "text": template.format(topic=topic),
            }
        )
    return posts


def summarize_recs(recs: list[dict]) -> dict:
    if not recs:
        return {"n": 0}
    stance_mean = sum(float(r.get("stance_score", 0.0)) for r in recs) / len(recs)
    strength_mean = sum(float(r.get("strength", 0.0)) for r in recs) / len(recs)
    topic_sim_mean = sum(float(r.get("topic_similarity", 0.0)) for r in recs) / len(recs)
    labels = [((r.get("metadata") or {}).get("label") or "?") for r in recs]
    modes = [r.get("mode", "?") for r in recs]
    return {
        "n": len(recs),
        "stance_mean": stance_mean,
        "strength_mean": strength_mean,
        "topic_sim_mean": topic_sim_mean,
        "labels": Counter(labels),
        "modes": Counter(modes),
    }


async def main():
    topics = load_topics(TOPICS_PATH)
    forced = "vaccine"
    topic = forced if forced else random.choice(topics)

    if topic == "vaccine":
        topic = "vaccines"

    analyzer = EmbeddingAnalyzer(topic, use_local_embedding_model=True)
    store = RollingEmbeddingStore(topic=topic, analyzer=analyzer, redis_cache=None)

    if topic == "vaccines":
        corpus = build_vaccines_corpus()
    else:
        corpus = build_generic_corpus(topic, n=30)

    for i, item in enumerate(corpus):
        await store.add(
            item["text"],
            id=f"post-{i}",
            metadata={
                "sender_id": item.get("sender_id"),
                "label": item.get("label"),
            },
        )

    # Multiple queries mimicking users with different stances/subtopics
    queries: list[str]
    if topic == "vaccines":
        queries = [
            "Vaccines save lives and the fear-mongering is out of control. We should boost access and fight misinformation.",
            "I don’t trust pharma and I’m against mandates. People deserve informed consent without coercion.",
            "I’m genuinely undecided. What are the real risks by age, and what benefits are we seeing?",
            "Mandates might make sense in hospitals, but blanket mandates feel wrong. What’s the balanced policy?",
            "I had a scary side effect and felt dismissed. How common are adverse reactions and what support exists?",
        ]
    else:
        queries = [
            f"I support {topic} and want it expanded quickly.",
            f"I oppose {topic}; the risks and costs are being ignored.",
            f"I’m neutral on {topic}. I want data and tradeoffs.",
        ]

    print(f"Topic: {topic}")
    print(f"Corpus size: {len(corpus)}")

    # Some weak sanity checks (skip if embeddings are not available)
    if not bool(os.getenv("OPENAI_API_KEY")):
        print("\nWARNING: OPENAI_API_KEY not set; embeddings may fail at runtime.")

    for qi, query in enumerate(queries):
        # Embed the query to get agent parameters
        embedded = await analyzer.embed_and_score(query, include_vector=True)
        if embedded is None:
            print(f"Failed to embed query {qi+1}, skipping")
            continue
        agent_vector = embedded["vector"]
        agent_stance = embedded["stance_score"]
        agent_strength = embedded["strength"]

        recs = await store.recommend_for_agent_vector(
            agent_vector=agent_vector,
            agent_stance_score=agent_stance,
            agent_strength=agent_strength,
            top_k=10,
            diversity_prob=0.5,
        )
        summary = summarize_recs(recs)

        print("\n" + "=" * 80)
        print(f"Query {qi+1}: {query}")
        print(
            "Top-10 summary: "
            f"n={summary['n']} stance_mean={summary.get('stance_mean', 0.0):.3f} "
            f"strength_mean={summary.get('strength_mean', 0.0):.3f} "
            f"topic_sim_mean={summary.get('topic_sim_mean', 0.0):.3f} "
            f"labels={dict(summary.get('labels', {}))} modes={dict(summary.get('modes', {}))}"
        )

        if not recs:
            raise RuntimeError("No recommendations returned; corpus may be empty or filtered.")

        print("\nTop recommendations:")
        for r in recs:
            meta = r.get("metadata") or {}
            label = meta.get("label", "?")
            sender = meta.get("sender_id", "?")
            print(
                f"- d={r['distance']:.3f} stance={r['stance_score']:.3f} strength={r['strength']:.3f} "
                f"mode={r.get('mode', '?')} label={label} sender={sender} :: {r['text']}"
            )

        # Weak directional checks for vaccines (should hold in most runs)
        # if topic == "vaccines":
        #     stance_mean = float(summary.get("stance_mean", 0.0))
        #     if qi == 1:
        #         assert stance_mean > -0.2, "Pro-vaccine query unexpectedly got strongly negative stance recs"
        #     if qi == 2:
        #         assert stance_mean < 0.2, "Anti-mandate query unexpectedly got strongly positive stance recs"


if __name__ == "__main__":
    asyncio.run(main())
