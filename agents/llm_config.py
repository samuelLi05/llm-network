from __future__ import annotations

import os
import threading
from typing import Optional, Any

from dotenv import load_dotenv
from openai import OpenAI


_OLLAMA_MAX_CONCURRENCY = max(1, int(os.getenv("OLLAMA_MAX_CONCURRENCY", "1")))
_OLLAMA_CALL_SEMAPHORE = threading.Semaphore(_OLLAMA_MAX_CONCURRENCY)


class _OllamaChatCompletionsProxy:
    def __init__(self, client: OpenAI) -> None:
        self._client = client

    def create(self, *args: Any, **kwargs: Any) -> Any:
        # Limit concurrent Ollama calls to avoid server overload/500s.
        _OLLAMA_CALL_SEMAPHORE.acquire()
        try:
            #extra_body = kwargs.pop("extra_body", {}) or {}

            # Inject think=false unless explicitly provided
            #extra_body.setdefault("think", False)
            return self._client.chat.completions.create(*args, extra_body=extra_body, **kwargs)
        finally:
            _OLLAMA_CALL_SEMAPHORE.release()


class _OllamaChatProxy:
    def __init__(self, client: OpenAI) -> None:
        self._completions = _OllamaChatCompletionsProxy(client)

    @property
    def completions(self) -> _OllamaChatCompletionsProxy:
        return self._completions


class _OllamaClientWrapper:
    def __init__(self, client: OpenAI) -> None:
        self._client = client
        self._chat = _OllamaChatProxy(client)

    @property
    def chat(self) -> _OllamaChatProxy:
        return self._chat

load_dotenv()


# global instances initialized by main.py via initialize_active_client()
ACTIVE_OPENAI_CLIENT: Optional[Any] = None
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
        base_client = OpenAI(
            api_key=ollama_api_key,
            base_url=ollama_base_url,
        )
        ACTIVE_OPENAI_CLIENT = _OllamaClientWrapper(base_client)
        ACTIVE_OPENAI_MODEL = ollama_model
        return

    ACTIVE_OPENAI_CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=openai_base_url)
    ACTIVE_OPENAI_MODEL = openai_model
