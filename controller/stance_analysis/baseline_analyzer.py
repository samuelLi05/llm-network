import os
import asyncio
import math
from typing import Optional
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
from peft import PeftModel
from agents.local_llm import HuggingFaceLLM as LocalLLM
from agents.llm_service import LLMService

# Load in API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

"""Analyzes the stance of text prompts using LLMs and SBERT."""
class BaselineAnalyzer:
    def __init__(self, topic:str, local_llm: Optional[LocalLLM] = None, llm_service: Optional[LLMService] = None):
        self.topic = topic
        self.init_prompt = f"You are an expert stance analyzer for the topic: {self.topic}. Analyze the stance of given texts."
        self.local_llm = local_llm
        self.llm_service = llm_service
        self.sbert_stance_analysis_model = SentenceTransformer('all-mpnet-base-v2')
        self.sbert_stance_analysis_model[0].auto_model = PeftModel.from_pretrained(
            self.sbert_stance_analysis_model[0].auto_model,
            'vahidthegreat/StanceAware-SBERT',
            device_map='auto'
        )

    async def openai_get_log_prob_classification(self, prompt:str) -> Optional[tuple[str, float, float]]:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                client.responses.create,
                model="gpt-4o-mini",
                input=[
                    {"role": "system", "content": self.init_prompt},
                    {"role": "user", "content": (
                        "Classify the following user prompt into one of these options: "
                        f"'A. Strongly Pro {self.topic}', "
                        f"'B. Moderately Pro {self.topic}', "
                        "'C. Neutral', "
                        f"'D. Moderately Anti {self.topic}', "
                        f"'E. Strongly Anti {self.topic}'. "
                        "Respond with only the single letter (A, B, C, D, or E).\n\n"
                        "User Prompt: " + prompt
                    )},
                ],
                temperature=0.4,
                max_output_tokens=16,
                top_logprobs=20,
                include=["message.output_text.logprobs"],
                text={"format": {"type": "text"}},
            ),
            timeout=60
        )

        logprob_items = None
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant":
                for part in getattr(item, "content", []) or []:
                    if getattr(part, "type", None) == "output_text":
                        logprob_items = getattr(part, "logprobs", None)
                        if logprob_items:
                            break
            if logprob_items:
                break

        if not logprob_items:
            return None

        first = logprob_items[0]
        top_logprobs = getattr(first, "top_logprobs", None) or []
        if not top_logprobs:
            return None

        logprob_map = {}
        for item in top_logprobs:
            token = getattr(item, "token", "")
            if not token:
                continue
            key = token.strip()
            label = key[:1]
            if label in {"A", "B", "C", "D", "E"}:
                logprob_map[label] = max(logprob_map.get(label, float("-inf")), item.logprob)

        prob_a = math.exp(logprob_map.get("A", float("-inf")))
        prob_b = math.exp(logprob_map.get("B", float("-inf")))
        prob_c = math.exp(logprob_map.get("C", float("-inf")))
        prob_d = math.exp(logprob_map.get("D", float("-inf")))
        prob_e = math.exp(logprob_map.get("E", float("-inf")))

        total = prob_a + prob_b + prob_c + prob_d + prob_e
        if not total:
            return None

        probs = {
            "A": prob_a / total,
            "B": prob_b / total,
            "C": prob_c / total,
            "D": prob_d / total,
            "E": prob_e / total,
        }

        label = max(probs, key=probs.get)
        weights = {"A": 1.0, "B": 0.5, "C": 0.0, "D": -0.5, "E": -1.0}
        stance_score = sum(probs[k] * weights[k] for k in probs)
        return label, stance_score, probs[label]
    
    async def local_llm_classification(self, prompt:str) -> Optional[tuple[str, float, float]]:
        if self.llm_service:
            result = await self.llm_service.classify_stance(self.init_prompt, prompt, topic=self.topic)
            return result["label"], result["stance_score"], result["chosen_prob"]
        if self.local_llm:
            result = await asyncio.to_thread(
                self.local_llm.classify_stance,
                self.init_prompt,
                prompt,
                self.topic,
            )
            return result["label"], result["stance_score"], result["chosen_prob"]
        return None
        

    def local_sbert_find_similarity(self, sentnces: list[str]) -> list[list[float]]:
        # Use SBERT here for a simliarty check for opinion mining using pre trained hugging face model
        embeddings = self.sbert_stance_analysis_model.encode(
            sentnces,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        similarity_matrix = util.cos_sim(embeddings, embeddings)
        return similarity_matrix.cpu().tolist()