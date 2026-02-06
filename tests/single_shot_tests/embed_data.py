""" Model fitting and plotting the results for single-shot prediction tests. """

import matplotlib.pyplot as plt
import numpy as np
import json
import time
from dataclasses import dataclass
from typing import Any, Optional
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import sklearn
import csv
import os
import re
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file, including OPENAI_API_KEY

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

BASELINE_STATEMENT = "Vaccines cause autism"
BASELINE_TOPIC = "vaccine safety and autism"

# If True: mimic the new workflow direction where stance-aware SBERT drives
# topic/strength scoring, and OpenAI embeddings are used for cosine similarity.
USE_LOCAL_EMBEDDING_MODEL = True

# How to score output stance/topic/strength from social-media-style responses.
# Options: "heading" (first line), "weighted" (softmax-weighted mean over spans), "best_span" (single best span), "full" (full response)
SCORE_SPAN_MODE = "heading"

RAW_DATA_DIRECTORY = "tests/single_shot_tests/data/raw_text_jsonl"
CLEANED_DATA_DIRECTORY = "tests/single_shot_tests/data/stance_converted"

sbert_stance_analysis_model = SentenceTransformer('all-mpnet-base-v2')
sbert_stance_analysis_model[0].auto_model = PeftModel.from_pretrained(sbert_stance_analysis_model[0].auto_model, 'vahidthegreat/StanceAware-SBERT')

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client: Optional[OpenAI] = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

@dataclass(frozen=True)
class _AnchorFrame:
    subject: str
    as_statement: bool
    pro: np.ndarray
    anti: np.ndarray
    neutral: np.ndarray
    subject_vec: np.ndarray
    neg_subject_vec: np.ndarray
    axis: np.ndarray


_FRAME_CACHE: dict[tuple[str, bool], _AnchorFrame] = {}


def _anchor_texts(subject: str, *, as_statement: bool) -> dict[str, list[str]]:
    if not as_statement:
        return {
            "pro": [
                f"{subject} is the right direction. We should support it and move faster.",
                f"If you care about progress, you should back {subject}.",
                f"I’m in favor of {subject}. It’s practical and necessary.",
            ],
            "anti": [
                f"{subject} sounds good on paper, but it’s harmful in the real world.",
                f"I oppose {subject}. The costs and risks are being ignored.",
                f"We should push back on {subject}. It’s a mistake.",
            ],
            "neutral": [
                f"On {subject}, I see arguments on both sides. I’m still weighing tradeoffs.",
                f"{subject} has benefits and costs. I’m open to evidence.",
                f"Not convinced either way on {subject}. Let’s focus on facts, not hype.",
            ],
        }

    return {
        "pro": [
            f"{subject} is true.",
            f"It is a fact that {subject}.",
            f"The claim '{subject}' is correct.",
        ],
        "anti": [
            f"{subject} is false.",
            f"There is no good evidence that {subject}.",
            f"The claim '{subject}' is misinformation.",
        ],
        "neutral": [
            f"I'm not sure whether {subject}.",
            f"I haven't seen enough evidence to decide if {subject}.",
            f"I'm undecided about whether {subject}.",
        ],
    }


def _encode(texts: list[str]) -> np.ndarray:
    # SentenceTransformers returns unit-normalized vectors when normalize_embeddings=True.
    vectors = sbert_stance_analysis_model.encode(
        list(texts),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vectors.astype(np.float32, copy=False)


def _encode_openai(texts: list[str]) -> Optional[np.ndarray]:
    if openai_client is None:
        return None

    resp = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=list(texts),
    )
    vecs = np.asarray([item.embedding for item in resp.data], dtype=np.float32)
    # L2-normalize so cosine == dot
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    vecs = vecs / norms
    return vecs.astype(np.float32, copy=False)


def _get_frame(subject: str, *, as_statement: bool) -> _AnchorFrame:
    key = (str(subject), bool(as_statement))
    cached = _FRAME_CACHE.get(key)
    if cached is not None:
        return cached

    groups = _anchor_texts(subject, as_statement=as_statement)
    group_names = ["pro", "anti", "neutral"]
    anchor_texts: list[str] = []
    slices: dict[str, slice] = {}
    start = 0
    for name in group_names:
        texts = groups[name]
        end = start + len(texts)
        slices[name] = slice(start, end)
        anchor_texts.extend(texts)
        start = end

    extra = [subject]
    if as_statement:
        extra.append(f"It is not true that {subject}.")

    all_vecs = _encode(anchor_texts + extra)
    anchor_vecs = all_vecs[:-1]
    subject_vec = all_vecs[len(anchor_texts)]
    neg_subject_vec = all_vecs[len(anchor_texts) + 1] if as_statement else subject_vec

    def _centroid(name: str) -> np.ndarray:
        centroid = anchor_vecs[slices[name]].mean(axis=0)
        # re-normalize centroid
        norm = float(np.linalg.norm(centroid))
        if norm > 0:
            centroid = centroid / norm
        return centroid.astype(np.float32, copy=False)

    pro = _centroid("pro")
    anti = _centroid("anti")
    neutral = _centroid("neutral")

    axis = pro - anti
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm > 0:
        axis = axis / axis_norm
    axis = axis.astype(np.float32, copy=False)

    # Ensure axis sign: pro should be positive, anti negative.
    if float(np.dot(pro, axis)) < float(np.dot(anti, axis)):
        axis = (-axis).astype(np.float32, copy=False)

    frame = _AnchorFrame(
        subject=str(subject),
        as_statement=bool(as_statement),
        pro=pro,
        anti=anti,
        neutral=neutral,
        subject_vec=subject_vec,
        neg_subject_vec=neg_subject_vec,
        axis=axis,
    )
    _FRAME_CACHE[key] = frame
    return frame


def _score(vec: np.ndarray, frame: _AnchorFrame) -> dict[str, Any]:
    # vec is expected unit-normalized.
    sim_neutral = float(np.dot(vec, frame.neutral))
    subject_similarity = float(np.dot(vec, frame.subject_vec))
    if frame.as_statement:
        stance_score = float(subject_similarity - float(np.dot(vec, frame.neg_subject_vec)))
    else:
        sim_pro = float(np.dot(vec, frame.pro))
        sim_anti = float(np.dot(vec, frame.anti))
        stance_score = float(sim_pro - sim_anti)

    if USE_LOCAL_EMBEDDING_MODEL:
        strength = max(0.0, subject_similarity)
    else:
        strength = max(0.0, subject_similarity) * (1.0 - max(0.0, sim_neutral))

    return {
        # Keep names aligned with EmbeddingAnalyzer output.
        "topic_similarity": float(subject_similarity),
        "stance_score": float(stance_score),
        "strength": float(strength),
    }


def _split_spans(text: str, *, max_spans: int = 12) -> list[str]:
    t = (text or "").strip()
    if not t:
        return []
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    head = lines[0] if lines else t

    sent_parts = re.split(r"(?<=[.!?])\s+|\n+", t)
    sents = [s.strip() for s in sent_parts if s and s.strip()]

    candidates: list[str] = []
    for cand in [head] + lines[:3] + sents[:8]:
        c = (cand or "").strip()
        if not c or len(c) < 12:
            continue
        if len(c) > 300:
            c = c[:300].rsplit(" ", 1)[0] or c[:300]
        if c not in candidates:
            candidates.append(c)

    if len(sents) >= 2:
        combo2 = (sents[0] + " " + sents[1]).strip()
        if 12 <= len(combo2) <= 300 and combo2 not in candidates:
            candidates.append(combo2)
    if len(sents) >= 3:
        combo3 = (sents[0] + " " + sents[1] + " " + sents[2]).strip()
        if 12 <= len(combo3) <= 300 and combo3 not in candidates:
            candidates.append(combo3)

    return candidates[: int(max_spans)]


def _pick_best_output_span(
    *,
    response: str,
    in_vec: np.ndarray,
    frame: _AnchorFrame,
    stance_weight: float = 0.25,
    input_sim_weight: float = 0.15,
    max_spans: int = 12,
) -> tuple[str, np.ndarray]:
    """Pick the most topic-relevant, stance-bearing span from a social response."""
    spans = _split_spans(response, max_spans=max_spans)
    if not spans:
        fallback = (response.splitlines()[0].strip() if response else "") or response
        vec = _encode([fallback or ""])[0]
        return (fallback, vec)

    span_vecs = _encode(spans)
    best_i = 0
    best_score = float("-inf")

    for i, v in enumerate(span_vecs):
        scored = _score(v, frame)
        topic_sim = float(scored["topic_similarity"])
        stance = float(scored["stance_score"])
        sim_to_input = float(np.dot(in_vec, v))
        score = topic_sim + stance_weight * abs(stance) + input_sim_weight * sim_to_input
        if score > best_score:
            best_score = score
            best_i = i

    return (spans[best_i], span_vecs[best_i])


def _weighted_output_vec(
    *,
    response: str,
    in_vec: np.ndarray,
    frame: _AnchorFrame,
    stance_weight: float = 0.25,
    input_sim_weight: float = 0.15,
    temperature: float = 0.05,
    min_topic_similarity: float = 0.10,
    max_spans: int = 12,
) -> np.ndarray:
    """Softmax-weighted mean of span vectors (more robust than argmax)."""
    spans = _split_spans(response, max_spans=max_spans)
    if not spans:
        fallback = (response.splitlines()[0].strip() if response else "") or response
        return _encode([fallback or ""])[0]

    span_vecs = _encode(spans)

    scores: list[float] = []
    kept: list[int] = []
    for i, v in enumerate(span_vecs):
        s = _score(v, frame)
        topic_sim = float(s["topic_similarity"])
        if topic_sim < float(min_topic_similarity):
            continue
        stance_mag = abs(float(s["stance_score"]))
        sim_to_input = float(np.dot(in_vec, v))
        score = topic_sim + float(stance_weight) * stance_mag + float(input_sim_weight) * sim_to_input
        scores.append(float(score))
        kept.append(i)

    if not kept:
        return span_vecs[0]

    temp = float(temperature) if float(temperature) > 0 else 0.05
    mx = max(scores)
    exps = np.exp((np.asarray(scores, dtype=np.float32) - mx) / temp)
    denom = float(exps.sum()) or 1.0
    w = (exps / denom).astype(np.float32)

    agg = np.zeros_like(span_vecs[kept[0]])
    for weight, idx in zip(w, kept):
        agg += float(weight) * span_vecs[idx]

    # Re-normalize so cosine == dot.
    norm = float(np.linalg.norm(agg))
    if norm > 0:
        agg = agg / norm
    return agg.astype(np.float32, copy=False)


# Load data from RAW_DATA_DIRECTORY
def load_data():
    data = []
    if os.path.exists(RAW_DATA_DIRECTORY):
        for filename in os.listdir(RAW_DATA_DIRECTORY):
            if filename.endswith('.jsonl'):
                filepath = os.path.join(RAW_DATA_DIRECTORY, filename)
                with open(filepath, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data.append(json.loads(line))
    return data

# Clean data by converting to vector mebeddings and store result. 
def clean_data(
    *,
    include_vectors: bool = False,
    max_rows: Optional[int] = None,
    save_csv: bool = True,
) -> list[dict[str, Any]]:
    raw_data = load_data()
    if max_rows is not None:
        raw_data = raw_data[: int(max_rows)]
    topic_frame = _get_frame(BASELINE_STATEMENT, as_statement=True)

    cleaned_data: list[dict[str, Any]] = []
    os.makedirs(CLEANED_DATA_DIRECTORY, exist_ok=True)

    for item in raw_data:
        stance_sentence = str(item.get("stance_sentence") or "")
        response = str(item.get("response") or "")

        # Stance-aware SBERT embeddings (used for topic/strength scoring).
        # - similarity uses the full response
        # - stance/topic/strength for output uses the best sentence/span selected by scoring
        in_vec_sbert, out_vec_sbert_full = _encode([stance_sentence, response])
        mode = str(SCORE_SPAN_MODE or "heading").strip().lower()
        if mode == "weighted":
            out_vec_sbert_score = _weighted_output_vec(
                response=response,
                in_vec=in_vec_sbert,
                frame=topic_frame,
                max_spans=12,
            )
        elif mode == "best_span":
            _best_span_text, out_vec_sbert_score = _pick_best_output_span(
                response=response,
                in_vec=in_vec_sbert,
                frame=topic_frame,
                max_spans=12,
            )
        elif mode == "full":
            out_vec_sbert_score = _encode([response or ""])[0]
        else:
            # Default: first line / heading
            response_head = (response.splitlines()[0].strip() if response else "") or response
            out_vec_sbert_score = _encode([response_head or ""])[0]
        in_out_similarity_sbert = float(np.dot(in_vec_sbert, out_vec_sbert_full))

        # OpenAI embeddings (used only for cosine similarity between vectors).
        openai_vecs = _encode_openai([stance_sentence, response])
        if openai_vecs is None:
            in_vec_openai = out_vec_openai = None
            in_out_similarity_openai = None
        else:
            in_vec_openai, out_vec_openai = openai_vecs
            in_out_similarity_openai = float(np.dot(in_vec_openai, out_vec_openai))

        input_topic = _score(in_vec_sbert, topic_frame)
        output_topic = _score(out_vec_sbert_score, topic_frame)

        record: dict[str, Any] = {
            "topic": item.get("topic") or BASELINE_TOPIC,
            "stance_weight": item.get("stance_weight"),
            "features": {
                "in_out_similarity_openai": in_out_similarity_openai,
                "in_out_similarity_sbert": in_out_similarity_sbert,
                "input": input_topic,
                "output": output_topic,
            }
        }

        if include_vectors:
            record["vectors"] = {
                "input_sbert": in_vec_sbert.astype(np.float32, copy=False).tolist(),
                "output_sbert": out_vec_sbert_full.astype(np.float32, copy=False).tolist(),
                "input_openai": (in_vec_openai.astype(np.float32, copy=False).tolist() if in_vec_openai is not None else None),
                "output_openai": (out_vec_openai.astype(np.float32, copy=False).tolist() if out_vec_openai is not None else None),
            }

        cleaned_data.append(record)

    if save_csv:
        out_path = os.path.join(
            CLEANED_DATA_DIRECTORY,
            f"cleaned_pairs_{time.strftime('%Y%m%d-%H%M%S')}.csv",
        )
        headers = [
            "topic",
            "stance_weight",
            "in_out_similarity_openai",
            "in_out_similarity_sbert",
            "input_topic_similarity",
            "input_stance_score",
            "input_strength",
            "output_topic_similarity",
            "output_stance_score",
            "output_strength"
        ]
        if include_vectors:
            headers.extend(["input_sbert_vector", "output_sbert_vector", "input_openai_vector", "output_openai_vector"])

        with open(out_path, "w", encoding="utf-8", newline="") as csvf:
            writer = csv.writer(csvf)
            writer.writerow(headers)
            for row in cleaned_data:
                feat = row.get("features", {})
                in_feat = feat.get("input", {})
                out_feat = feat.get("output", {})
                values = [
                    row.get("topic"),
                    row.get("stance_weight"),
                    feat.get("in_out_similarity_openai"),
                    feat.get("in_out_similarity_sbert"),
                    in_feat.get("topic_similarity"),
                    in_feat.get("stance_score"),
                    in_feat.get("strength"),
                    out_feat.get("topic_similarity"),
                    out_feat.get("stance_score"),
                    out_feat.get("strength")
                ]
                if include_vectors:
                    v = row.get("vectors", {})
                    values.append(json.dumps(v.get("input_sbert"), ensure_ascii=False))
                    values.append(json.dumps(v.get("output_sbert"), ensure_ascii=False))
                    values.append(json.dumps(v.get("input_openai"), ensure_ascii=False))
                    values.append(json.dumps(v.get("output_openai"), ensure_ascii=False))
                writer.writerow(values)

    return cleaned_data


if __name__ == "__main__":
    clean_data(include_vectors=False, save_csv=True)