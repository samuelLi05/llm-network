"""
Instantiate a runner without the network for single shot opinion drift testing and prediction
"""

import asyncio
import json
import os
import random
import sys
import time
from datetime import datetime, timezone

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from logs.logger import console_logger

from agents.network_agent import NetworkAgent
from agents.llm_service import LLMService
from agents.local_llm import HuggingFaceLLM
from agents.prompt_configs.generate_prompt import PromptGenerator

USE_LOCAL_LLM = True
# Total prompts = len(WEIGHTS) * SAMPLES_PER_WEIGHT * REPEATS_PER_PROMPT
SAMPLES_PER_WEIGHT = 5
REPEATS_PER_PROMPT = 4
MAX_CONCURRENCY = 1
BASELINE_ASSIGNMENT_SEED = 42

USE_BASELINE_STATEMENT = True
BASELINE_STATEMENT = "Vaccines cause autism"
BASELINE_TOPIC = "vaccine safety and autism"

WEIGHTS = [-1.0, -0.5, 0.0, 0.5, 1.0]

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "raw_text_jsonl")

initial_prompt_template = (
   "You are participating in a social-media-style discussion about {topic}." \
   "The sentence, {unique_prompt}, is your fixed stance and is authoritative and exhaustive. Write entirely from the worldview, assumptions, tone, values, and constraints it defines; it fully determines what you believe, how you speak, and what claims you are willing to make." \
   "Produce a short, attention-grabbing post that hooks readers, makes a clear and strong claim aligned with that grounding, and invites engagement (likes, replies, shares)." \
   "Be concise, bold, and evocative. Use a distinct memorable opening line, assertive language, and a direct call-to-action every time. Emulate authentic social media posts." \
   "Make sure posts are distinct, do not copy formatting and language of previous posts, instead contradict any claims that oppose your fixed stance"
   "Do not introduce outside viewpoints, neutral framing, balance, or meta-commentary. Do not soften or qualify claims unless explicitly required by the authoritative sentence. Never refer to yourself as an agent, AI, or participant in a debate."
)


async def generate_data():
    console_logger.info("Starting LLM Network...")

    llm_service = None
    if USE_LOCAL_LLM:
        console_logger.info("Using local LLM service (quantized HF model).")
        local_llm = HuggingFaceLLM()
        llm_service = LLMService(local_llm)
        await llm_service.start()
    else:
        local_llm = None

    # 1. Initialize shared components
    prompt_generator = PromptGenerator()
    os.makedirs(DATA_DIR, exist_ok=True)

    # We'll write JSONL so multiple runs can be concatenated safely.
    out_name = f"single_shot_pairs_{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.jsonl"
    out_path = os.path.join(DATA_DIR, out_name)

    if USE_BASELINE_STATEMENT:
        topic = BASELINE_TOPIC
        console_logger.info(f"Baseline mode enabled: topic='{topic}' stance='{BASELINE_STATEMENT}'")

        # Base pool from PromptGenerator (5 templates => 5 points).
        base_fixed_pool = prompt_generator.generate_fixed_opinions(BASELINE_STATEMENT, weighted_values=WEIGHTS)
        base_items = list(base_fixed_pool.items())  # (stance_sentence, weight)
        if len(base_items) != len(WEIGHTS):
            raise RuntimeError(f"Expected {len(WEIGHTS)} baseline templates, got {len(base_items)}")

        # Expand beyond 5 points by adding more stance sentence phrasings per weight.
        extra_by_weight: dict[float, list[str]] = {
            -1.0: [
                f"I reject the claim that {BASELINE_STATEMENT}.",
                f"The idea that {BASELINE_STATEMENT} is false and dangerous misinformation.",
                f"I am firmly against the statement: {BASELINE_STATEMENT}.",
            ],
            -0.5: [
                f"I’m skeptical of the claim that {BASELINE_STATEMENT}.",
                f"I doubt that {BASELINE_STATEMENT} is accurate.",
                f"I question the statement: {BASELINE_STATEMENT}.",
            ],
            0.0: [
                f"I’m undecided about whether {BASELINE_STATEMENT}.",
                f"I’m not convinced either way that {BASELINE_STATEMENT}.",
                f"I remain neutral on the statement: {BASELINE_STATEMENT}.",
            ],
            0.5: [
                f"I tend to agree that {BASELINE_STATEMENT}.",
                f"I think there may be truth to the idea that {BASELINE_STATEMENT}.",
                f"I lean toward believing the statement: {BASELINE_STATEMENT}.",
            ],
            1.0: [
                f"I strongly believe that {BASELINE_STATEMENT}.",
                f"I’m certain the statement is true: {BASELINE_STATEMENT}.",
                f"There’s no doubt in my mind that {BASELINE_STATEMENT}.",
            ],
        }

        pool_by_weight: dict[float, list[str]] = {w: [] for w in WEIGHTS}
        for stance_sentence, w in base_items:
            pool_by_weight[float(w)].append(stance_sentence)
        for w in WEIGHTS:
            pool_by_weight[w].extend(extra_by_weight.get(w, []))

        rng = random.Random(BASELINE_ASSIGNMENT_SEED)
        samples: list[dict] = []
        for w in WEIGHTS:
            stance_sentences = list(dict.fromkeys(pool_by_weight[w]))  # stable de-dupe
            rng.shuffle(stance_sentences)
            # Cycle if SAMPLES_PER_WEIGHT > number of templates.
            for j in range(SAMPLES_PER_WEIGHT):
                stance_sentence = stance_sentences[j % len(stance_sentences)]
                init_prompt = initial_prompt_template.format(topic=topic, unique_prompt=stance_sentence)
                samples.append(
                    {
                        "stance_weight": float(w),
                        "stance_sentence": stance_sentence,
                        "init_prompt": init_prompt,
                    }
                )

        rng.shuffle(samples)
    else:
        topic = prompt_generator.get_topic()
        console_logger.info(f"Shared discussion topic: {topic}")

        # Non-baseline mode: just sample prompts (no labels/weights).
        raw_prompts = prompt_generator.generate_multiple_prompts(len(WEIGHTS) * SAMPLES_PER_WEIGHT)
        samples = []
        for stance_sentence in raw_prompts:
            init_prompt = initial_prompt_template.format(topic=topic, unique_prompt=stance_sentence)
            samples.append(
                {
                    "stance_weight": None,
                    "stance_sentence": stance_sentence,
                    "init_prompt": init_prompt,
                }
            )

    console_logger.info(f"Prepared {len(samples)} initialization prompts.")

    sem = asyncio.Semaphore(int(MAX_CONCURRENCY))

    async def _run_one(idx: int, sample: dict, repeat_i: int) -> dict:
        agent_id = f"agent_{idx+1}_r{repeat_i+1}"
        agent = NetworkAgent(
            id=agent_id,
            init_prompt=str(sample["init_prompt"]),
            topic=topic,
            llm_service=llm_service,
        )
        async with sem:
            response = await agent.generate_response()
        return {
            "ts": time.time(),
            "topic": topic,
            "baseline_statement": BASELINE_STATEMENT if USE_BASELINE_STATEMENT else None,
            "agent_id": agent_id,
            "stance_weight": sample.get("stance_weight"),
            "stance_sentence": sample.get("stance_sentence"),
            "response": response,
        }

    records: list[dict] = []
    tasks = []
    for i, sample in enumerate(samples):
        for r in range(int(REPEATS_PER_PROMPT)):
            tasks.append(_run_one(i, sample, r))

    # Run sequentially if MAX_CONCURRENCY=1, otherwise bounded parallelism.
    for coro in tasks:
        rec = await coro
        records.append(rec)
        console_logger.info(
            f"Wrote sample agent={rec['agent_id']} weight={rec['stance_weight']} chars={len(rec.get('response') or '')}"
        )

        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    console_logger.info(f"Dataset written: {out_path} (rows={len(records)})")

    if llm_service is not None:
        await llm_service.stop()


if __name__ == "__main__":
    asyncio.run(generate_data())