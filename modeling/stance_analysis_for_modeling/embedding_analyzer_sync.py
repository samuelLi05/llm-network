
from typing import Optional
from sentence_transformers import SentenceTransformer
from controller.stance_analysis.vector_ops import l2_normalize, dot_similarity_normalized
from peft import PeftModel

class EmbeddingAnalyzerSync:

    def __init__(
            self,
            topic: str, 
            use_local_embedding_model: bool = True,
            use_baseline_statement: bool = True,
            local_embedding_model: str = "all-mpnet-base-v2",
    ):
        self.topic = topic
        self.use_local_embedding_model = bool(use_local_embedding_model)
        self.use_baseline_statement = bool(use_baseline_statement)
        self.local_embedding_model = str(local_embedding_model)


        if not self.use_baseline_statement:
            raise NotImplementedError("Non-baseline statement embedding is not yet implemented. ")

        self._topic_embedding: Optional[list[float]] = None
        self._neg_topic_embedding: Optional[list[float]] = None


        # initialize the local embedding model if specified
        if self.use_local_embedding_model:
            model = SentenceTransformer(self.local_embedding_model)
            model[0].auto_model = PeftModel.from_pretrained(
                model[0].auto_model, "vahidthegreat/StanceAware-SBERT"
            )
            self._st_model = model
            self._st_model_name = self.local_embedding_model
        else:
            raise NotImplementedError("Remote embedding service is not yet implemented. ")

    def _ensure_anchor_embeddings(self) -> None:
        if (
            self._topic_embedding is not None 
            and self._neg_topic_embedding is not None
        ):
            return
        
        texts: list[str] = [self.topic]
        if self.use_baseline_statement:
            texts.append(f"It is not true that {self.topic}.")

        vectors = self._embed_texts_for_scoring(texts)
        topic_vec = vectors[0]
        neg_topic_vec = vectors[1] 

        self._topic_embedding = l2_normalize(topic_vec)
        self._neg_topic_embedding = l2_normalize(neg_topic_vec)

    def _embed_texts_local(self, texts: list[str]) -> list[list[float]]:
        vectors = self._st_model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [v.astype("float32").tolist() for v in vectors]


    def _embed_texts_for_scoring(self, texts: list[str]) -> list[list[float]]:
        if self.use_local_embedding_model:
            return self._embed_texts_local(texts)
        else:
            raise NotImplementedError("Remote embedding service is not yet implemented. ")

    def score_vector(self, vector: list[float], *, include_vector: bool = False) -> Optional[dict]:
        self._ensure_anchor_embeddings()
        prompt_vec = l2_normalize(vector)

        topic_similarity = dot_similarity_normalized(prompt_vec, self._topic_embedding)
        if self.use_baseline_statement:
            stance_score = topic_similarity - dot_similarity_normalized(prompt_vec, self._neg_topic_embedding)
        else:
            raise NotImplementedError("Non-baseline statement scoring is not yet implemented. ")
        


        result = {
            "topic_similarity": topic_similarity,
            "neg_topic_similarity": dot_similarity_normalized(prompt_vec, self._neg_topic_embedding),
            "stance_score": stance_score,
            "model": self._st_model_name
            }
        if include_vector:
            result["vector"] = prompt_vec
        return result

    def embed_and_score(
        self,
        prompt: str, 
        *,
        score_text: Optional[str] = None,
    ):
        

        score_source = prompt if score_text is None else score_text
        score_vec = l2_normalize(self._embed_texts_for_scoring([score_source])[0])
        scored = self.score_vector(score_vec, include_vector=False)

        return scored