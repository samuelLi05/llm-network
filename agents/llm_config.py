from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


# global instances initialized by main.py via initialize_active_client()
ACTIVE_OPENAI_CLIENT: Optional[OpenAI] = None
ACTIVE_OPENAI_MODEL: Optional[str] = None


def initialize_active_client(
    llm_api_backend: str,
    openai_model: str,
    openai_base_url: Optional[str],
    ollama_model: str,
    ollama_base_url: str,
    ollama_api_key: str,
    openai_api_key: Optional[str],
) -> None:
    """Initialize the global OpenAI client singleton.
    
    All config values are passed by main.py, not read from module globals.
    """
    global ACTIVE_OPENAI_CLIENT, ACTIVE_OPENAI_MODEL

    backend = str(llm_api_backend).strip().lower()
    if backend == "ollama":
        ACTIVE_OPENAI_CLIENT = OpenAI(api_key=ollama_api_key, base_url=ollama_base_url)
        ACTIVE_OPENAI_MODEL = ollama_model
        return

    ACTIVE_OPENAI_CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=openai_base_url)
    ACTIVE_OPENAI_MODEL = openai_model
