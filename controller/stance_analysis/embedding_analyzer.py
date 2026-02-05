import os
import asyncio
import math
import threading
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from peft import PeftModel
from agents.local_llm import HuggingFaceLLM as LocalLLM
from agents.llm_service import LLMService

from controller.stance_analysis.vector_ops import dot_similarity_normalized, l2_normalize, mean_vector

# Load in API key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class EmbeddingAnalyzer:
    _shared_st_model = None
    _shared_st_model_name: Optional[str] = None
    _shared_st_lock = threading.Lock()

    def __init__(
        self,
        topic: str,
        local_llm: Optional[LocalLLM] = None,
        llm_service: Optional[LLMService] = None,
        *,
        use_openai_embeddings: bool = True,
        use_baseline_statement: bool = False,
        openai_embedding_model: str = "text-embedding-3-small",
        local_embedding_model: str = "all-mpnet-base-v2",
    ):
        self.topic = topic
        self.local_llm = local_llm
        self.llm_service = llm_service
        self.use_openai_embeddings = bool(use_openai_embeddings)
        self.use_baseline_statement = bool(use_baseline_statement)
        self.embedding_model = (
            str(openai_embedding_model) if self.use_openai_embeddings else str(local_embedding_model)
        )

        self._client: Optional[OpenAI] = OpenAI(api_key=OPENAI_API_KEY)

        self._st_model = None
        self.similarity_temperature = 1.0

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
        } if not self.use_baseline_statement else {
            "pro": [
                f"The statement, {self.topic}, is wholeheartedly true.",
                f"I strongly agree with the statement, {self.topic}.",
                f"Supporting the notion that {self.topic} is the right course of action.",
            ],
            "anti": [
                f"The statement, {self.topic}, has significant drawbacks and should be opposed.",
                f"I strongly disagree with the statement, {self.topic}.",
                f"Opposing the statement, {self.topic}, is necessary to avoid harm.",
            ],
            "neutral": [
                f"The statement, {self.topic}, has both pros and cons that need to be considered.",
                f"I remain neutral on the statement, {self.topic} until more evidence is available.",
                f"The statement, {self.topic}, requires further analysis before forming a strong opinion.",
            ],
        }

        self._anchor_group_embeddings: Optional[dict[str, list[float]]] = None
        self._topic_embedding: Optional[list[float]] = None

    async def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self.use_openai_embeddings:
            if self._client is None:
                raise RuntimeError("OpenAI client not initialized")
            response = await asyncio.to_thread(
                self._client.embeddings.create,
                model=self.embedding_model,
                input=texts,
            )
            return [item.embedding for item in response.data]

        # Local embeddings (SentenceTransformers)
        def _encode_local() -> list[list[float]]:
            if self._st_model is None:
                # Instantiate with thread lock
                with EmbeddingAnalyzer._shared_st_lock:
                    if (
                        EmbeddingAnalyzer._shared_st_model is None
                        or EmbeddingAnalyzer._shared_st_model_name != self.embedding_model
                    ):
                        model = SentenceTransformer(self.embedding_model)
                        model[0].auto_model = PeftModel.from_pretrained(
                            model[0].auto_model, "vahidthegreat/StanceAware-SBERT"
                        )
                        EmbeddingAnalyzer._shared_st_model = model
                        EmbeddingAnalyzer._shared_st_model_name = self.embedding_model
                    self._st_model = EmbeddingAnalyzer._shared_st_model

            vectors = self._st_model.encode(
                list(texts),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return [v.astype("float32").tolist() for v in vectors]

        return await asyncio.to_thread(_encode_local)

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

