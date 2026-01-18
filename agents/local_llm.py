from typing import Optional, List, Union, Sequence, Dict, Iterable
import os
import torch
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
)

class HuggingFaceLLM:
    def __init__(
        self,
        model_name: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        quantize: bool = True,
        use_4bit: bool = True,
        trust_remote_code: Optional[bool] = None,
    ):
        # Default to the valid HF repo for Qwen3-VL-8B Instruct
        self.model_name = model_name or os.getenv("LOCAL_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.quantize = quantize
        self.use_4bit = use_4bit
        if trust_remote_code is None:
            trust_remote_code = os.getenv("HF_TRUST_REMOTE_CODE", "true").lower() in {"1", "true", "yes"}
        self.trust_remote_code = trust_remote_code

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model '{self.model_name}' on device '{self.device}'...")

        try:
            model_cls = Qwen3VLForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", trust_remote_code=self.trust_remote_code)

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True,
                trust_remote_code=self.trust_remote_code,
            )
            if self.quantize and self.use_4bit:
                self.model = model_cls.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    trust_remote_code=self.trust_remote_code,
                )
            elif self.quantize and not self.use_4bit:
                self.model = model_cls.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    load_in_8bit=True,
                    trust_remote_code=self.trust_remote_code,
                )
            else:
                self.model = model_cls.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    low_cpu_mem_usage=True,
                    trust_remote_code=self.trust_remote_code,
                )
        except TypeError:
            # Fallback for older transformers without load_in_4bit/load_in_8bit
            self.model = model_cls.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True,
                trust_remote_code=self.trust_remote_code,
            )
        except OSError as exc:
            raise OSError(
                f"Failed to load model '{self.model_name}'. "
                "Set LOCAL_MODEL to a valid Hugging Face repo or local path. "
                "If the repo is private, run `hf auth login` or set HUGGINGFACE_HUB_TOKEN."
            ) from exc
        self.model.eval()

        # Required for some models (e.g., LLaMA)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(
        self,
        prompt: Union[str, Sequence[Dict[str, str]]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        max_tokens = max_new_tokens or self.max_new_tokens
        temperature = temperature or self.temperature
        messages = self._normalize_messages(prompt)
        inputs = self._build_inputs(messages, add_generation_prompt=True)

        # inference
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Remove prompt tokens
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Stop sequence handling
        if stop:
            for s in stop:
                if s in text:
                    text = text.split(s)[0]

        return text.strip()

    def score_label_logprob(
        self,
        prompt: Union[str, Sequence[Dict[str, str]]],
        label: str,
        normalize: bool = True,
    ) -> Dict[str, float]:
        """Compute log-probability of `label` given `prompt`.

        Returns a dict with raw log-prob, avg log-prob per token, and token length.
        """
        messages = self._normalize_messages(prompt)

        # Build prompt-only inputs
        prompt_inputs = self._build_inputs(messages, add_generation_prompt=True)
        prompt_len = prompt_inputs["input_ids"].shape[1]

        # Build full inputs with assistant label appended
        label_messages = list(messages) + [
            {"role": "assistant", "content": [{"type": "text", "text": label}]}
        ]
        full_inputs = self._build_inputs(label_messages, add_generation_prompt=False)

        with torch.no_grad():
            logits = self.model(**full_inputs).logits  # (1, seq_len, vocab)

        # Align logits to target tokens (shifted by one)
        input_ids = full_inputs["input_ids"][0]
        next_token_logits = logits[0, :-1, :]
        target_tokens = input_ids[1:]

        # Slice label tokens range
        start = max(prompt_len - 1, 0)
        end = min(start + (input_ids.shape[0] - prompt_len), next_token_logits.shape[0])
        label_logits = next_token_logits[start:end]
        label_targets = target_tokens[start:end]

        if label_logits.shape[0] == 0:
            return {"logprob": float("-inf"), "avg_logprob": float("-inf"), "num_tokens": 0.0}

        label_logps = F.log_softmax(label_logits, dim=-1).gather(
            1, label_targets.unsqueeze(-1)
        ).squeeze(-1)

        total_logprob = float(label_logps.sum().item())
        num_tokens = float(label_logps.shape[0])
        avg_logprob = float(label_logps.mean().item()) if normalize else total_logprob

        return {
            "logprob": total_logprob,
            "avg_logprob": avg_logprob,
            "num_tokens": num_tokens,
        }

    def _normalize_messages(
        self,
        prompt: Union[str, Sequence[Dict[str, str]]],
    ) -> List[Dict[str, Union[str, List[Dict[str, str]]]]]:
        """Normalize prompt into Qwen-VL style messages."""
        if isinstance(prompt, str):
            return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        messages: List[Dict[str, Union[str, List[Dict[str, str]]]]] = []
        for m in prompt:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):
                normalized_content = content
            else:
                normalized_content = [{"type": "text", "text": str(content)}]
            messages.append({"role": role, "content": normalized_content})
        return messages

    def _build_inputs(self, messages: Iterable[Dict[str, str]], add_generation_prompt: bool):
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in inputs.items()}
