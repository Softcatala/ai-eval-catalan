"""Shared inference parameters and small provider-specific conversions."""

from pathlib import Path

import yaml


INFERENCE_CONFIG = yaml.safe_load(
    Path(__file__).with_name("inference.yaml").read_text(encoding="utf-8")
)


def inference_params(max_tokens=None, provider="llama", model_name=""):
    params = {**INFERENCE_CONFIG}
    if max_tokens is not None:
        params["max_tokens"] = max_tokens

    model_name = model_name.lower()
    if provider == "gemini":
        if "gemini-3" in model_name:
            if params["reasoning_effort"] == "none":
                params["reasoning_effort"] = "minimal"
            params["max_tokens"] = max(params["max_tokens"], 1024)
        elif not model_name.startswith("gemini-"):
            params.pop("reasoning_effort")
    elif provider == "openai":
        if model_name.startswith("gpt-5."):
            params.pop("temperature")
            params["max_tokens"] = max(params["max_tokens"], 1024)
        else:
            params.pop("reasoning_effort")
    elif provider in ("openrouter", "hf"):
        params.pop("reasoning_effort")

    return params


def chat_completion_params(max_tokens=None, provider="llama", model_name=""):
    params = inference_params(max_tokens, provider, model_name)
    if provider != "llama":
        params["max_completion_tokens"] = params.pop("max_tokens")
    return params


def lm_eval_params(max_tokens, provider, model_name):
    params = inference_params(max_tokens, provider, model_name)
    if provider == "hf":
        params["max_gen_toks"] = params.pop("max_tokens")
    return params
