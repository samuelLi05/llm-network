import os
import asyncio
import math
from typing import Optional
from openai import OpenAI
from agents.local_llm import HuggingFaceLLM as LocalLLM
from agents.llm_service import LLMService

# Load in API key
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

    @staticmethod
    def _mean_vector(vectors: list[list[float]]) -> list[float]:
        if not vectors:
            return []
        dim = len(vectors[0])
        acc = [0.0] * dim
        for vec in vectors:
            for i, v in enumerate(vec):
                acc[i] += v
        n = float(len(vectors))
        return [v / n for v in acc]

    @staticmethod
    def _l2_normalize(vec: list[float]) -> list[float]:
        norm = math.sqrt(sum(v * v for v in vec))
        if not norm:
            return vec
        return [v / norm for v in vec]

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
            centroid = self._mean_vector(anchor_vectors[s])
            group_embeddings[name] = self._l2_normalize(centroid)

        self._anchor_group_embeddings = group_embeddings
        self._topic_embedding = self._l2_normalize(topic_vec)

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for a, b in zip(vec_a, vec_b):
            dot += a * b
            norm_a += a * a
            norm_b += b * b
        denom = math.sqrt(norm_a) * math.sqrt(norm_b)
        return dot / denom if denom else 0.0

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
        await self._ensure_anchor_embeddings()
        if not self._anchor_group_embeddings or self._topic_embedding is None:
            return None

        prompt_vec = self._l2_normalize((await self._embed_texts([prompt]))[0])

        sim_pro = self._cosine_similarity(prompt_vec, self._anchor_group_embeddings["pro"])
        sim_anti = self._cosine_similarity(prompt_vec, self._anchor_group_embeddings["anti"])
        sim_neutral = self._cosine_similarity(prompt_vec, self._anchor_group_embeddings["neutral"])
        topic_similarity = self._cosine_similarity(prompt_vec, self._topic_embedding)

        # Topic stance axis: project prompt onto (pro - anti) direction.
        axis = [a - b for a, b in zip(self._anchor_group_embeddings["pro"], self._anchor_group_embeddings["anti"]) ]
        axis = self._l2_normalize(axis)
        stance_score = self._cosine_similarity(prompt_vec, axis)

        # Strength: how opinionated w.r.t. this topic (high when far from neutral AND on-topic)
        strength = max(0.0, topic_similarity) * (1.0 - max(0.0, sim_neutral))

        return {
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

