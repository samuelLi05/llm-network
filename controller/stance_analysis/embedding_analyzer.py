import os
import asyncio
import math
import threading
import re
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
        use_local_embedding_model: bool = False,
        use_baseline_statement: bool = False,
        score_span_mode: str = "heading",
        score_span_temperature: float = 0.05,
        score_span_min_topic_similarity: float = 0.10,
        score_span_stance_weight: float = 0.25,
        openai_embedding_model: str = "text-embedding-3-small",
        local_embedding_model: str = "all-mpnet-base-v2",
    ):
        self.topic = topic
        self.local_llm = local_llm
        self.llm_service = llm_service
        self.use_local_embedding_model = bool(use_local_embedding_model)
        self.use_baseline_statement = bool(use_baseline_statement)

        # How to extract stance/topic/strength signal from social-media-style text.
        # - "heading": score first line only (fast + often best for punchy posts)
        # - "best_span": pick one best sentence/span
        # - "weighted": softmax-weighted mean over spans
        # - "full": score the full post string
        self.score_span_mode = str(score_span_mode or "heading").strip().lower()
        self.score_span_temperature = float(score_span_temperature)
        self.score_span_min_topic_similarity = float(score_span_min_topic_similarity)
        self.score_span_stance_weight = float(score_span_stance_weight)

        self.openai_embedding_model = str(openai_embedding_model)
        self.local_embedding_model = str(local_embedding_model)

        # Semantic vectors are preferred from OpenAI (when available).
        self.semantic_embedding_model = self.openai_embedding_model
        self.score_embedding_model = self.local_embedding_model if self.use_local_embedding_model else self.openai_embedding_model

        self._client: Optional[OpenAI] = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

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
                f"{self.topic} is true.",
                f"It is a fact that {self.topic}.",
                f"The claim '{self.topic}' is correct.",
            ],
            "anti": [
                f"{self.topic} is false.",
                f"There is no good evidence that {self.topic}.",
                f"The claim '{self.topic}' is misinformation.",
            ],
            "neutral": [
                f"I'm not sure whether {self.topic}.",
                f"I haven't seen enough evidence to decide if {self.topic}.",
                f"I'm undecided about whether {self.topic}.",
            ],
        }

        # Anchor embeddings are computed in the scoring model space.
        self._anchor_group_embeddings: Optional[dict[str, list[float]]] = None
        self._topic_embedding: Optional[list[float]] = None
        self._neg_topic_embedding: Optional[list[float]] = None

    async def _embed_texts_openai(self, texts: list[str]) -> list[list[float]]:
        if self._client is None:
            raise RuntimeError("OpenAI embeddings requested but OPENAI_API_KEY is not set")
        response = await asyncio.to_thread(
            self._client.embeddings.create,
            model=self.openai_embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    async def _embed_texts_local(self, texts: list[str]) -> list[list[float]]:
        # Local embeddings (SentenceTransformers + stance-aware adapter)
        def _encode_local() -> list[list[float]]:
            if self._st_model is None:
                with EmbeddingAnalyzer._shared_st_lock:
                    if (
                        EmbeddingAnalyzer._shared_st_model is None
                        or EmbeddingAnalyzer._shared_st_model_name != self.local_embedding_model
                    ):
                        model = SentenceTransformer(self.local_embedding_model)
                        model[0].auto_model = PeftModel.from_pretrained(
                            model[0].auto_model, "vahidthegreat/StanceAware-SBERT"
                        )
                        EmbeddingAnalyzer._shared_st_model = model
                        EmbeddingAnalyzer._shared_st_model_name = self.local_embedding_model
                    self._st_model = EmbeddingAnalyzer._shared_st_model

            vectors = self._st_model.encode(
                list(texts),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return [v.astype("float32").tolist() for v in vectors]

        return await asyncio.to_thread(_encode_local)

    async def _embed_texts_for_scoring(self, texts: list[str]) -> list[list[float]]:
        if self.use_local_embedding_model:
            return await self._embed_texts_local(texts)
        return await self._embed_texts_openai(texts)

    async def _embed_texts_for_semantic(self, texts: list[str]) -> list[list[float]]:
        # In local-embedding mode we still prefer OpenAI vectors for similarity checks.
        if self.use_local_embedding_model and self._client is not None:
            return await self._embed_texts_openai(texts)
        # Fallback: use scoring-space embeddings as semantic vectors.
        return await self._embed_texts_for_scoring(texts)

    async def _ensure_anchor_embeddings(self) -> None:
        if (
            self._anchor_group_embeddings is not None
            and self._topic_embedding is not None
            and (not self.use_baseline_statement or self._neg_topic_embedding is not None)
        ):
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

        # Anchors are always embedded in the scoring model space.
        extra_texts: list[str] = [self.topic]
        if self.use_baseline_statement:
            extra_texts.append(f"It is not true that {self.topic}.")

        vectors = await self._embed_texts_for_scoring(all_anchor_texts + extra_texts)
        anchor_vectors = vectors[: len(all_anchor_texts)]
        topic_vec = vectors[len(all_anchor_texts)]
        neg_topic_vec = vectors[len(all_anchor_texts) + 1] if self.use_baseline_statement else None

        group_embeddings: dict[str, list[float]] = {}
        for name in group_names:
            s = group_slices[name]
            centroid = mean_vector(anchor_vectors[s])
            group_embeddings[name] = l2_normalize(centroid)

        self._anchor_group_embeddings = group_embeddings
        self._topic_embedding = l2_normalize(topic_vec)
        if self.use_baseline_statement and neg_topic_vec is not None:
            self._neg_topic_embedding = l2_normalize(neg_topic_vec)

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

    async def embed_and_score(
        self,
        prompt: str,
        *,
        include_vector: bool = False,
        semantic_text: Optional[str] = None,
        score_text: Optional[str] = None,
        precomputed_score_vector: Optional[list[float]] = None,
    ) -> Optional[dict]:
        await self._ensure_anchor_embeddings()
        if not self._anchor_group_embeddings or self._topic_embedding is None:
            return None

        score_source = prompt if score_text is None else score_text
        semantic_source = prompt if semantic_text is None else semantic_text

        # 1) Score in scoring space.
        if precomputed_score_vector is not None:
            score_vec = l2_normalize(precomputed_score_vector)
        else:
            score_vec = l2_normalize((await self._embed_texts_for_scoring([score_source]))[0])
        scored = await self.score_vector(score_vec, include_vector=False)
        if scored is None:
            return None

        # 2) Provide semantic vector for similarity checks (optional).
        semantic_model_used = (
            self.openai_embedding_model
            if (self.use_local_embedding_model and self._client is not None)
            else self.score_embedding_model
        )
        scored["model"] = semantic_model_used
        scored["score_model"] = self.score_embedding_model

        # Always include the scoring vector when it differs from the semantic vector
        # (e.g., headline-only stance scoring), or when using a distinct local scoring model.
        if self.use_local_embedding_model or (score_text is not None) or (semantic_text is not None and semantic_source != score_source):
            scored["score_vector"] = score_vec

        if include_vector:
            semantic_vec = l2_normalize((await self._embed_texts_for_semantic([semantic_source]))[0])
            scored["vector"] = semantic_vec

        return scored

    @staticmethod
    def _split_into_spans(text: str, *, max_spans: int = 12) -> list[str]:
        """Best-effort split of a social post into candidate score spans.

        We keep this dependency-free (no nltk/spacy). The goal is to produce a
        small list of short-ish candidates likely to contain the core claim.
        """
        t = (text or "").strip()
        if not t:
            return []

        # Prefer line splits first (common in social posts).
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        head = lines[0] if lines else t

        # Sentence-ish splitting.
        # Split on .!? followed by whitespace, and also treat newlines as boundaries.
        sent_parts = re.split(r"(?<=[.!?])\s+|\n+", t)
        sents = [s.strip() for s in sent_parts if s and s.strip()]

        # Build a candidate set with a stable order.
        candidates: list[str] = []
        for cand in [head] + lines[:3] + sents[:8]:
            c = (cand or "").strip()
            if not c:
                continue
            if len(c) < 12:
                continue
            # Cap very long spans (often full thread dumps).
            if len(c) > 300:
                c = c[:300].rsplit(" ", 1)[0] or c[:300]
            if c not in candidates:
                candidates.append(c)

        # Add a couple short composites (first 2-3 sentences) which can help when
        # the stance is expressed over two short fragments.
        if len(sents) >= 2:
            combo2 = (sents[0] + " " + sents[1]).strip()
            if 12 <= len(combo2) <= 300 and combo2 not in candidates:
                candidates.append(combo2)
        if len(sents) >= 3:
            combo3 = (sents[0] + " " + sents[1] + " " + sents[2]).strip()
            if 12 <= len(combo3) <= 300 and combo3 not in candidates:
                candidates.append(combo3)

        return candidates[: int(max_spans)]

    async def select_best_score_span(
        self,
        text: str,
        *,
        max_spans: int = 12,
        stance_weight: float = 0.25,
    ) -> tuple[str, list[float]]:
        """Select the best span for stance/topic/strength scoring.

        Returns (span_text, span_score_vector).
        Selection is based primarily on topic similarity (aboutness), with a
        small bonus for having strong polarity (|stance_score|).
        """
        await self._ensure_anchor_embeddings()
        if not self._anchor_group_embeddings or self._topic_embedding is None:
            return ("", [])

        spans = self._split_into_spans(text, max_spans=max_spans)
        if not spans:
            fallback = (text or "").strip()
            if not fallback:
                return ("", [])
            vec = (await self._embed_texts_for_scoring([fallback]))[0]
            return (fallback, vec)

        vectors = await self._embed_texts_for_scoring(spans)

        best_i = 0
        best_score = float("-inf")
        for i, vec in enumerate(vectors):
            v = l2_normalize(vec)
            # Inline scoring (avoids N calls to score_vector())
            topic_similarity = dot_similarity_normalized(v, self._topic_embedding)
            sim_pro = dot_similarity_normalized(v, self._anchor_group_embeddings["pro"])
            sim_anti = dot_similarity_normalized(v, self._anchor_group_embeddings["anti"])
            if self.use_baseline_statement and self._neg_topic_embedding is not None:
                stance_score = topic_similarity - dot_similarity_normalized(v, self._neg_topic_embedding)
            else:
                stance_score = sim_pro - sim_anti

            score = float(topic_similarity) + float(stance_weight) * abs(float(stance_score))
            if score > best_score:
                best_score = score
                best_i = i

        return (spans[best_i], vectors[best_i])

    async def aggregate_score_vector(
        self,
        text: str,
        *,
        max_spans: int = 12,
        stance_weight: float = 0.25,
        temperature: float = 0.05,
        min_topic_similarity: float = 0.10,
    ) -> Optional[list[float]]:
        """Return a softmax-weighted mean scoring vector over candidate spans.

        This is more robust than selecting a single best sentence for volatile
        social posts. Weighting focuses on spans that are (a) on-topic and (b)
        stance-bearing.
        """
        await self._ensure_anchor_embeddings()
        if not self._anchor_group_embeddings or self._topic_embedding is None:
            return None

        spans = self._split_into_spans(text, max_spans=max_spans)
        if not spans:
            t = (text or "").strip()
            if not t:
                return None
            vec = (await self._embed_texts_for_scoring([t]))[0]
            return vec

        vectors = await self._embed_texts_for_scoring(spans)
        topic_sims: list[float] = []
        stance_mags: list[float] = []
        keep: list[int] = []

        for i, vec in enumerate(vectors):
            v = l2_normalize(vec)
            topic_similarity = dot_similarity_normalized(v, self._topic_embedding)
            if topic_similarity < float(min_topic_similarity):
                continue

            sim_pro = dot_similarity_normalized(v, self._anchor_group_embeddings["pro"])
            sim_anti = dot_similarity_normalized(v, self._anchor_group_embeddings["anti"])
            if self.use_baseline_statement and self._neg_topic_embedding is not None:
                stance_score = topic_similarity - dot_similarity_normalized(v, self._neg_topic_embedding)
            else:
                stance_score = sim_pro - sim_anti

            keep.append(i)
            topic_sims.append(float(topic_similarity))
            stance_mags.append(abs(float(stance_score)))

        if not keep:
            # If everything fell below threshold, fall back to the first span.
            return vectors[0]

        # Softmax over a combined score.
        temp = float(temperature) if float(temperature) > 0 else 0.05
        logits = [ts + float(stance_weight) * sm for ts, sm in zip(topic_sims, stance_mags)]
        max_logit = max(logits)
        exps = [math.exp((l - max_logit) / temp) for l in logits]
        denom = sum(exps) or 1.0
        weights = [e / denom for e in exps]

        # Weighted mean in scoring space.
        dim = len(vectors[keep[0]])
        agg = [0.0] * dim
        for w, idx in zip(weights, keep):
            vec = vectors[idx]
            for j in range(dim):
                agg[j] += w * float(vec[j])
        return l2_normalize(agg)

    async def score_vector(self, vector: list[float], *, include_vector: bool = False) -> Optional[dict]:
        """Score an already-embedded vector against this topic's anchor frame.

        This is the key for precomputed agent profiles: no embedding call required.
        The input vector should already be L2-normalized.
        """
        await self._ensure_anchor_embeddings()
        if not self._anchor_group_embeddings or self._topic_embedding is None:
            return None

        # Note: `vector` is expected to be in the scoring model space.
        prompt_vec = l2_normalize(vector)

        sim_pro = dot_similarity_normalized(prompt_vec, self._anchor_group_embeddings["pro"])
        sim_anti = dot_similarity_normalized(prompt_vec, self._anchor_group_embeddings["anti"])
        sim_neutral = dot_similarity_normalized(prompt_vec, self._anchor_group_embeddings["neutral"])
        topic_similarity = dot_similarity_normalized(prompt_vec, self._topic_embedding)

        if self.use_baseline_statement and self._neg_topic_embedding is not None:
            stance_score = topic_similarity - dot_similarity_normalized(prompt_vec, self._neg_topic_embedding)
        else:
            # Default: difference in similarity to pro vs anti anchors.
            stance_score = sim_pro - sim_anti

        if self.use_local_embedding_model:
            strength = max(0.0, topic_similarity)
        else:
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
            # `model` is filled in by embed_and_score(); for score_vector() we expose the scoring model.
            "model": self.score_embedding_model,
        }

        if include_vector:
            result["vector"] = prompt_vec

        return result

