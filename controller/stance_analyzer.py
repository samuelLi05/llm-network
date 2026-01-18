import os
import asyncio
from typing import Optional, TYPE_CHECKING
from dotenv import load_dotenv
from openai import OpenAI
from network.stream import RedisStream
from agents.local_llm import HuggingFaceLLM as LocalLLM
from agents.llm_service import LLMService


class StanceAnalyzer:
    def __init__(self):
        pass

    def openai_get_log_prob_classification(self):
        pass

    def eval_log_probs(self):
        pass

    def find_similarity(self):
        # Use SBERT here for a simliarty check for opinion mining using pre trained hugging face model
        pass
