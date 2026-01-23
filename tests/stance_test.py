import asyncio
import json
import os
import sys
import random
from typing import Dict, List, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from controller.stance_analysis.baseline_analyzer import BaselineAnalyzer
from agents.local_llm import HuggingFaceLLM
from agents.llm_service import LLMService

from sentence_transformers import SentenceTransformer, util

TOPICS_PATH = os.path.join(
	os.path.dirname(__file__),
	"..",
	"agents",
	"prompt_configs",
	"random_prompt.json",
)

STANCE_LABELS = {
	"A": "Pro",
	"B": "Neutral",
	"C": "Anti",
}

SOCIAL_TEMPLATES = {
	"A": [
		"{topic} is a win for everyday people. If you care about progress, you should stand behind it—no excuses.",
		"I’m all-in on {topic}. It’s practical, necessary, and the right direction. Who’s with me?",
		"Let’s stop pretending: {topic} is the future. Support it loudly and clearly.",
	],
	"B": [
		"On {topic}, I see valid points on multiple sides. I’m still weighing the tradeoffs.",
		"{topic} has benefits and costs. I’m open to evidence before taking a hard stance.",
		"Not convinced either way on {topic}. Let’s focus on facts, not hype.",
	],
	"C": [
		"{topic} sounds good on paper, but in reality it’s harmful and short-sighted. We should push back.",
		"I oppose {topic}. It’s a bad idea with real-world costs that people keep ignoring.",
		"Let’s be honest: {topic} is a mistake. We need to change course.",
	],
}


def load_topics(path: str) -> List[str]:
	with open(path, "r", encoding="utf-8") as f:
		data = json.load(f)
	return data.get("topics", [])


def generate_social_post(topic: str, stance_label: str) -> str:
	template = random.choice(SOCIAL_TEMPLATES[stance_label])
	return template.format(topic=topic)


def format_probs(probs: Dict[str, float]) -> str:
	return ", ".join(f"{k}={v:.3f}" for k, v in probs.items())


async def run_topic_tests(
	topic: str,
	use_openai: bool,
	use_local: bool,
	local_llm: HuggingFaceLLM | None,
	llm_service: LLMService | None,
	sbert_model: SentenceTransformer | None,
):
	print(f"\n=== Topic: {topic} ===")

	analyzer = BaselineAnalyzer(topic, local_llm=local_llm, llm_service=llm_service)

	num_per_stance = int(os.getenv("NUM_POSTS_PER_STANCE", "2"))
	samples: List[Tuple[str, str]] = []
	for label in ["A", "B", "C"]:
		for _ in range(num_per_stance):
			samples.append((label, generate_social_post(topic, label)))
	random.shuffle(samples)

	prototypes = None
	if sbert_model and util:
		prototypes = {
			"A": f"I strongly support {topic}.",
			"B": f"I feel neutral about {topic}.",
			"C": f"I strongly oppose {topic}.",
		}

	for idx, (true_label, text) in enumerate(samples, start=1):
		print(f"\n[{idx}] True stance: {true_label} ({STANCE_LABELS[true_label]})")
		print(f"Post: {text}")

		if use_openai:
			try:
				label, stance_score, chosen_prob = await analyzer.openai_get_log_prob_classification(text)
				print(
					"OpenAI  => "
					f"label={label}, stance_score={stance_score:.3f}, chosen_prob={chosen_prob:.3f}"
				)
			except Exception as exc:
				print(f"OpenAI  => ERROR: {exc}")

		if use_local:
			try:
				result = await analyzer.local_llm_classification(text)
				if result is None:
					print("Local   => SKIPPED (no local model)")
				else:
					label, stance_score, chosen_prob = result
					print(
						"Local   => "
						f"label={label}, stance_score={stance_score:.3f}, chosen_prob={chosen_prob:.3f}"
					)
			except Exception as exc:
				print(f"Local   => ERROR: {exc}")

		if prototypes and sbert_model and util:
			try:
				texts = [text] + list(prototypes.values())
				emb = sbert_model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
				sims = util.cos_sim(emb[0], emb[1:]).squeeze(0).tolist()
				sim_map = {k: float(v) for k, v in zip(prototypes.keys(), sims)}
				print("SBERT   => " + format_probs(sim_map))
			except Exception as exc:
				print(f"SBERT   => ERROR: {exc}")

	# Pairwise similarity of the generated posts (optional)
	try:
		similarity_matrix = analyzer.local_sbert_find_similarity([s[1] for s in samples])
		if similarity_matrix is not None:
			print("\nPairwise similarity matrix (SBERT):")
			for row in similarity_matrix:
				print("  " + " ".join(f"{v:0.2f}" for v in row))
	except Exception as exc:
		print(f"Pairwise similarity => ERROR: {exc}")


async def main():
	random.seed(7)
	topics = load_topics(TOPICS_PATH)
	if not topics:
		print("No topics found. Check random_prompt.json.")
		return

	num_topics = int(os.getenv("NUM_TOPICS", "3"))
	chosen_topics = random.sample(topics, k=min(num_topics, len(topics)))

	use_openai = bool(os.getenv("OPENAI_API_KEY"))
	use_local = True

	local_llm = None
	llm_service = None

	if use_local:
		local_llm = HuggingFaceLLM()
		llm_service = LLMService(local_llm)
		await llm_service.start()

	sbert_model = SentenceTransformer("all-mpnet-base-v2") if SentenceTransformer else None

	print("=== Stance Analyzer Smoke Test ===")
	print(f"OpenAI enabled: {use_openai}")
	print(f"Local  enabled: {use_local}")
	print(f"SBERT  enabled: {bool(sbert_model)}")

	for topic in chosen_topics:
		await run_topic_tests(
			topic,
			use_openai=use_openai,
			use_local=use_local,
			local_llm=local_llm,
			llm_service=llm_service,
			sbert_model=sbert_model,
		)

	if llm_service:
		await llm_service.stop()


if __name__ == "__main__":
	asyncio.run(main())
