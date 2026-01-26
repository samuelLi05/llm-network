import os
import asyncio
import math
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv
from agents.local_llm import HuggingFaceLLM as LocalLLM
from agents.llm_service import LLMService

from controller.stance_analysis.vector_ops import dot_similarity_normalized, l2_normalize, mean_vector

# Load in API key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

class EmbeddingAnalyzer:
    def __init__(self, topic:str, local_llm: Optional[LocalLLM] = None, llm_service: Optional[LLMService] = None):
        self.topic = topic
        self.local_llm = local_llm
        self.llm_service = llm_service
        self.embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.similarity_temperature = float(os.getenv("EMBEDDING_SIM_TEMPERATURE", "1.0"))

        # Anchors are NOT "labels"; they are fixed reference points for a topic-specific
        # coordinate system (pro/anti/neutral) that you can use to score and rank items.
        # These are written in a social-media style similar to your generators.
        self.anchor_groups: dict[str, list[str]] = {
            "pro": [
                f"{self.topic} is the right direction. We should support it and move faster.",
                f"If you care about progress, you should back {self.topic}.",
                f"I’m in favor of {self.topic}. It’s practical and necessary.",
            ],
            "anti": [
                f"{self.topic} sounds good on paper, but it’s harmful in the real world.",
                f"I oppose {self.topic}. The costs and risks are being ignored.",
                f"We should push back on {self.topic}. It’s a mistake.",
            ],
            "neutral": [
                f"On {self.topic}, I see arguments on both sides. I’m still weighing tradeoffs.",
                f"{self.topic} has benefits and costs. I’m open to evidence.",
                f"Not convinced either way on {self.topic}. Let’s focus on facts, not hype.",
            ],
        }

        self._anchor_group_embeddings: Optional[dict[str, list[float]]] = None
        self._topic_embedding: Optional[list[float]] = None

    async def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = await asyncio.to_thread(
            client.embeddings.create,
            model=self.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    async def _ensure_anchor_embeddings(self) -> None:
        if self._anchor_group_embeddings is not None and self._topic_embedding is not None:
            return

        group_names = list(self.anchor_groups.keys())
        all_anchor_texts: list[str] = []
        group_slices: dict[str, slice] = {}
        start = 0
        for name in group_names:
            texts = self.anchor_groups[name]
            end = start + len(texts)
            group_slices[name] = slice(start, end)
            all_anchor_texts.extend(texts)
            start = end

        vectors = await self._embed_texts(all_anchor_texts + [self.topic])
        anchor_vectors = vectors[:-1]
        topic_vec = vectors[-1]

        group_embeddings: dict[str, list[float]] = {}
        for name in group_names:
            s = group_slices[name]
            centroid = mean_vector(anchor_vectors[s])
            group_embeddings[name] = l2_normalize(centroid)

        self._anchor_group_embeddings = group_embeddings
        self._topic_embedding = l2_normalize(topic_vec)

    def _softmax(self, scores: list[float]) -> list[float]:
        if not scores:
            return []
        temperature = self.similarity_temperature if self.similarity_temperature > 0 else 1.0
        scaled = [v / temperature for v in scores]
        max_score = max(scaled)
        exp_scores = [math.exp(v - max_score) for v in scaled]
        total = sum(exp_scores)
        if not total:
            return [0.0 for _ in scores]
        return [v / total for v in exp_scores]

    async def embedding_classification(self, prompt: str) -> Optional[dict]:
        return await self.embed_and_score(prompt, include_vector=False)

    async def embed_and_score(self, prompt: str, *, include_vector: bool = False) -> Optional[dict]:
        await self._ensure_anchor_embeddings()
        if not self._anchor_group_embeddings or self._topic_embedding is None:
            return None

        prompt_vec = l2_normalize((await self._embed_texts([prompt]))[0])
        return await self.score_vector(prompt_vec, include_vector=include_vector)

    async def score_vector(self, vector: list[float], *, include_vector: bool = False) -> Optional[dict]:
        """Score an already-embedded vector against this topic's anchor frame.

        This is the key for precomputed agent profiles: no embedding call required.
        The input vector should already be L2-normalized.
        """
        await self._ensure_anchor_embeddings()
        if not self._anchor_group_embeddings or self._topic_embedding is None:
            return None

        prompt_vec = l2_normalize(vector)

        sim_pro = dot_similarity_normalized(prompt_vec, self._anchor_group_embeddings["pro"])
        sim_anti = dot_similarity_normalized(prompt_vec, self._anchor_group_embeddings["anti"])
        sim_neutral = dot_similarity_normalized(prompt_vec, self._anchor_group_embeddings["neutral"])
        topic_similarity = dot_similarity_normalized(prompt_vec, self._topic_embedding)

        # Topic stance axis: project prompt onto (pro - anti) direction.
        axis = [a - b for a, b in zip(self._anchor_group_embeddings["pro"], self._anchor_group_embeddings["anti"]) ]
        axis = l2_normalize(axis)
        stance_score = dot_similarity_normalized(prompt_vec, axis)

        # Strength: how opinionated w.r.t. this topic (high when far from neutral AND on-topic)
        strength = max(0.0, topic_similarity) * (1.0 - max(0.0, sim_neutral))

        result = {
            "topic_similarity": topic_similarity,
            "stance_score": stance_score,
            "strength": strength,
            "anchor_group_similarities": {
                "pro": sim_pro,
                "anti": sim_anti,
                "neutral": sim_neutral,
            },
            "model": self.embedding_model,
        }

        if include_vector:
            result["vector"] = prompt_vec

        return result

