from typing import Optional, List, Union, Sequence, Dict
import os
import torch
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

        # Normalize prompt into Qwen-VL style messages
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        else:
            messages = []
            for m in prompt:
                role = m.get("role", "user")
                content = m.get("content", "")
                if isinstance(content, list):
                    # Assume content already in multimodal format
                    normalized_content = content
                else:
                    # Wrap plain string content for Qwen-VL processor
                    normalized_content = [{"type": "text", "text": str(content)}]
                messages.append({"role": role, "content": normalized_content})

        # Prepare for inference using the model's chat template
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Move tensors to device (inputs is a dict-like BatchEncoding)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

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
