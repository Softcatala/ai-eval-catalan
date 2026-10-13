"""Native Anthropic inference through Bedrock Runtime with bearer authentication."""

import os

import requests

from inference import call_with_retries


def runtime_url(base_url, model):
    return f"{base_url.rstrip('/')}/model/{model}/invoke"


class BedrockRuntimeModel:
    def __init__(self, api_key, model_name, base_url):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url

    def generate(self, prompt, max_new_tokens=None):
        def invoke():
            response = requests.post(
                runtime_url(self.base_url, self.model_name),
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": max(max_new_tokens or 2048, 2048),
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=300,
            )
            response.raise_for_status()
            return "".join(
                block["text"]
                for block in response.json()["content"]
                if block.get("type") == "text"
            ).strip()

        return call_with_retries(invoke, "Bedrock Runtime invocation")


def runtime_lm_model(model, base_url):
    from lm_eval.models.anthropic_llms import AnthropicChat

    class RuntimeAnthropicChat(AnthropicChat):
        @property
        def header(self):
            return {"Authorization": f"Bearer {os.environ['AWS_BEARER_TOKEN_BEDROCK']}"}

        def _create_payload(self, *args, **kwargs):
            payload = super()._create_payload(*args, **kwargs)
            payload.pop("model", None)
            payload.pop("temperature", None)
            payload.pop("reasoning_effort", None)
            payload["anthropic_version"] = "bedrock-2023-05-31"
            payload["max_tokens"] = max(payload.get("max_tokens", 2048), 2048)
            return payload

        @staticmethod
        def parse_generations(outputs, **kwargs):
            if not isinstance(outputs, list):
                outputs = [outputs]
            return [
                "".join(b["text"] for b in output["content"] if b.get("type") == "text")
                for output in outputs
            ]

    return RuntimeAnthropicChat(
        model=model,
        base_url=runtime_url(base_url, model),
        tokenizer_backend="none",
        tokenized_requests=False,
        num_concurrent=1,
        max_retries=3,
        timeout=300,
    )
