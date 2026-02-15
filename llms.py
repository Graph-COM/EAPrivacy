# llms.py
import os
import json
import asyncio
from typing import Union, List, Dict, Any, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI
from tqdm.asyncio import tqdm_asyncio
import time
from transformers import AutoTokenizer
from vllm import LLM as VLLM, SamplingParams
import torch

# --- Model‐to‐Provider Mapping ---
# To add a new model:
# 1. Add the model name as a key and the provider ("google", "openai", "vllm") as the value in SUPPORTED_MODELS.
# 2. If the model requires specific configuration (e.g., thinking budget), add it to the model_budget_map in LLM_General.__init__.
SUPPORTED_MODELS = {
    # Google Models
    "gemini-2.5-flash-lite": "google",
    "gemini-2.5-flash-lite-w.o.think": "google",
    "gemini-2.5-flash": "google",
    "gemini-2.5-flash-w.o.think": "google",
    "gemini-2.5-pro": "google",
    "gemini-2.5-pro-w.o.think": "google",
    # OpenAI Models
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai",
    "gpt-5-low": "openai",
    "gpt-5-high": "openai",
    # Open-Source Models (via vLLM)
    # These keys should match the model name or path used by vLLM
    "unsloth.Llama-3.3-70B-Instruct": "vllm",
    "Qwen.Qwen3-30B-A3B": "vllm",
    "Qwen.Qwen3-32B": "vllm",
}


def get_llm(
    model_name: str,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    **kwargs: Any,
) -> "LLM_General":
    """
    Factory to get an LLM instance.  You can pass temperature & max_tokens here
    (they become defaults on generate()).
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model '{model_name}'.  "
            f"Supported: {list(SUPPORTED_MODELS)}"
        )
    provider = SUPPORTED_MODELS[model_name]
    return LLM_General(
        model_name=model_name,
        provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )


class LLM_General:
    """
    Unified interface for Google, OpenAI, and vLLM.
    Single‐turn generation with optional temperature/max_tokens.
    """

    def __init__(
        self,
        model_name: str,
        provider: str = "google",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = None
        self.thinking_budget = None
        self.reasoning_effort = None
        self.tokenizer = None

        # Specific budget assignments
        # Add any model-specific configurations here
        # The key is the user-facing model name (from SUPPORTED_MODELS),
        # and the value is a tuple of (actual_api_model_name, thinking_budget).
        model_budget_map = {
            "gemini-2.5-flash-lite-w.o.think": ("gemini-2.5-flash-lite", 0),
            "gemini-2.5-flash-lite": ("gemini-2.5-flash-lite", 24576),
            "gemini-2.5-pro-w.o.think": ("gemini-2.5-pro", 128),
            "gemini-2.5-flash-w.o.think": ("gemini-2.5-flash", 0),
        }
        if self.model_name in model_budget_map:
            self.model_name, self.thinking_budget = model_budget_map[self.model_name]

        # Handle GPT-5 reasoning efforts
        if self.model_name == "gpt-5-low":
            self.model_name = "openai.gpt-5"
            self.reasoning_effort = "low"
        elif self.model_name == "gpt-5-high":
            self.model_name = "openai.gpt-5"
            self.reasoning_effort = "high"

        load_dotenv()  # pick up any .env keys

        if provider == "google":
            api_key = (
                kwargs.get("api_key")
                or os.getenv("GEMINI_API_KEY")
                or os.getenv("GOOGLE_API_KEY")
            )
            if not api_key or api_key == "YOUR_API_KEY":
                raise ValueError(
                    "Set GEMINI_API_KEY or GOOGLE_API_KEY in your environment."
                )
            self.client = genai.Client(api_key=api_key)

        elif provider == "openai":
            api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
            base_url = kwargs.get("base_url") or os.getenv("OPENAI_ENDPOINT")
            if not api_key or api_key == "YOUR_API_KEY":
                raise ValueError("Set OPENAI_API_KEY in your environment.")
            self.client = OpenAI(api_key=api_key, base_url=base_url)

        elif provider == "vllm":
            # Get the number of available GPUs
            num_gpus = torch.cuda.device_count()
            if num_gpus == 0:
                raise ValueError("vLLM requires at least one GPU.")

            # Helper to clean up model names for vLLM path resolution
            if "qwen" in self.model_name.lower() and self.model_name.lower().endswith(
                "-thinking"
            ):
                base_model_name = self.model_name.replace("-thinking", "")
                model_name = base_model_name.replace(".", "/", 1)
            elif "oss" in self.model_name.lower():
                model_name = "openai/gpt-oss-120b"
            else:
                model_name = self.model_name.replace(".", "/", 1)

            if (
                "qwen" in self.model_name.lower()
                or "oss" in self.model_name.lower()
                or "llama" in self.model_name.lower()
            ):
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name, trust_remote_code=True
                )
                self.max_tokens = 10240
            self.client = VLLM(
                model=model_name,
                tensor_parallel_size=num_gpus,
                max_model_len=self.max_tokens,
                gpu_memory_utilization=0.9,
            )

        else:
            raise ValueError(f"Unknown provider '{provider}'.")

    def generate(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: int = 3,
    ) -> str:
        """
        Single turn generation.  Retries silently if we see 'error' in output
        or an exception is thrown.  Returns last error text if all retries fail.
        """
        if temperature is None:
            temperature = self.temperature
        if max_tokens is None:
            max_tokens = self.max_tokens

        last_error = ""
        for _ in range(max_retries):
            try:
                response_text = ""
                if self.provider == "google":
                    # Serialize conversational prompts to a single string for Google
                    if isinstance(prompt, list):
                        prompt_str = "\n".join(
                            f"{msg['role']}: {msg['content']}" for msg in prompt
                        )
                    else:
                        prompt_str = prompt

                    config = {}
                    if temperature is not None:
                        config["temperature"] = temperature
                    if max_tokens is not None:
                        config["max_output_tokens"] = max_tokens
                    if self.thinking_budget is not None:
                        config["thinking_config"] = types.ThinkingConfig(
                            thinking_budget=self.thinking_budget
                        )
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt_str,
                        config=config,
                    )
                    response_text = response.text

                elif self.provider == "openai":
                    if isinstance(prompt, str):
                        messages = [{"role": "user", "content": prompt}]
                    else:
                        messages = prompt
                    kwargs = {}
                    # if temperature is not None:
                    #     kwargs["temperature"] = temperature
                    # if max_tokens is not None:
                    #     kwargs["max_tokens"] = max_tokens

                    if self.reasoning_effort:
                        kwargs["extra_body"] = {
                            "reasoning_effort": self.reasoning_effort
                        }

                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        **kwargs,
                    )
                    response_text = completion.choices[0].message.content

                elif self.provider == "vllm":
                    sampling_params = SamplingParams(
                        temperature=0.6, top_p=0.95, top_k=20, min_p=0
                    )
                    # vLLM expects a list of prompts
                    if "qwen" in self.model_name.lower():
                        prompts = [
                            {
                                "role": "system",
                                "content": "You are a helpful assistant.",
                            },
                            {
                                "role": "user",
                                "content": (prompt),
                            },
                        ]
                        prompts = self.tokenizer.apply_chat_template(
                            prompts, tokenize=False, add_generation_prompt=True
                        )

                    else:
                        prompts = [prompt] if isinstance(prompt, str) else prompt
                    outputs = self.client.generate(prompts, sampling_params)
                    response_text = outputs[0].outputs[0].text

                else:
                    return f"Error: provider '{self.provider}' not supported."

                # if the provider spat out an error, retry
                if "error" in response_text.lower():
                    last_error = response_text
                    continue
                # otherwise return the clean text
                return response_text

            except Exception as e:
                last_error = f"[{self.provider} generation error] {e}"

        # all retries failed
        return last_error

    async def generate_async(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: int = 3,
    ) -> str:
        """
        Asynchronous version of generate.
        """
        return await asyncio.to_thread(
            self.generate,
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )


async def generate_concurrently(
    llm: "LLM_General",
    prompts: List[Union[str, List[Dict[str, str]]]],
    max_concurrency: int = 1,
) -> List[str]:
    """
    Generates responses for a list of prompts concurrently with a limit.
    For vLLM, it uses the optimized batch generation function.
    """
    if llm.provider == "vllm":
        return await generate_batch_vllm(llm, prompts)
    semaphore = asyncio.Semaphore(max_concurrency)

    async def process_prompt(prompt):
        async with semaphore:
            return await llm.generate_async(prompt)

    tasks = [process_prompt(prompt) for prompt in prompts]
    responses = await tqdm_asyncio.gather(
        *tasks, desc=f"Evaluating Prompts for {llm.model_name}"
    )
    time.sleep(1)
    return responses


async def generate_batch_vllm(
    llm: "LLM_General",
    prompts: List[Union[str, List[Dict[str, str]]]],
) -> List[str]:
    """
    Generates responses for a list of prompts using vLLM's batch processing.
    If the input length is bigger than max_tokens, cut it.
    """
    if llm.provider != "vllm":
        raise ValueError("This function is only for vLLM models.")
    if "qwen" in llm.model_name.lower():
        kwargs = {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0,
            "max_tokens": llm.max_tokens,
        }
    elif "oss" in llm.model_name.lower():
        kwargs = {"temperature": 1, "top_p": 1.0, "max_tokens": llm.max_tokens}
    else:
        kwargs = {
            "temperature": 0,
            "max_tokens": llm.max_tokens,
        }
    sampling_params = SamplingParams(
        **kwargs,
    )
    processed_prompts = []
    for prompt in prompts:
        if "oss" in llm.model_name.lower():
            effort = ""
            if isinstance(prompt, list):
                messages = list(prompt)  # make a copy
                if messages and messages[-1]["role"] == "assistant":
                    effort = messages.pop()["content"]
            else:  # string
                messages = [{"role": "user", "content": str(prompt)}]

            # now, we split the last reasoning effort
            if effort:
                effort = effort.split("Reasoning:")[-1].strip()

            system_prompt = {
                "role": "system",
                "content": f"""You are ChatModal, a large language model trained by Modal.
                Knowledge cutoff: 2025-08
                Reasoning: {effort}""",
            }
            chat_prompt = [system_prompt] + messages
            processed_prompt = llm.tokenizer.apply_chat_template(
                chat_prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
        elif (
            llm.tokenizer
        ):  # For Qwen, Llama, and any other vLLM model with a tokenizer
            if isinstance(prompt, str):
                chat_prompt = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant.",
                    },
                    {"role": "user", "content": prompt},
                ]
            else:
                chat_prompt = prompt

            template_kwargs = {}
            if "qwen" in llm.model_name.lower():
                template_kwargs["enable_thinking"] = "thinking" in llm.model_name

            processed_prompt = llm.tokenizer.apply_chat_template(
                chat_prompt,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
        else:
            # Fallback for models without a tokenizer (expects string prompts)
            if isinstance(prompt, list):
                last_user_message = next(
                    (
                        msg["content"]
                        for msg in reversed(prompt)
                        if msg["role"] == "user"
                    ),
                    str(prompt),
                )
                processed_prompt = last_user_message
            else:
                processed_prompt = prompt

        # --- CUT PROMPT IF TOO LONG ---
        if llm.tokenizer:
            input_ids = llm.tokenizer(processed_prompt, return_tensors="pt").input_ids[
                0
            ]
            if len(input_ids) > llm.max_tokens:
                # Truncate tokens and decode back to string
                truncated_ids = input_ids[: llm.max_tokens]
                processed_prompt = llm.tokenizer.decode(
                    truncated_ids, skip_special_tokens=True
                )
        else:
            # For string prompts, just cut the string if too long (approximate)
            if llm.max_tokens and (
                isinstance(processed_prompt, str)
                and len(processed_prompt) > llm.max_tokens * 4
            ):
                processed_prompt = processed_prompt[: llm.max_tokens * 4]

        processed_prompts.append(processed_prompt)

    mini_batch = 40
    outputs = []
    for i in range(0, len(processed_prompts), mini_batch):
        batch = processed_prompts[i : i + mini_batch]
        batch_outputs = llm.client.generate(batch, sampling_params)
        outputs.extend(batch_outputs)
        await asyncio.sleep(0)

    return [output.outputs[0].text for output in outputs]
