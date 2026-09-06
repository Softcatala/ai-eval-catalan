import json
import urllib.request
from pathlib import Path

from inference import call_with_retries, chat_completion_params, inference_params


_MUSE_USER_TEMPLATE = Path(__file__).with_name("templates") / "muse_glimmer_user.jinja"
_MUSE_BOS_TOKEN = "<|begin_of_text|>"
_AYA_END_OF_TURN_TOKEN = "<|END_OF_TURN_TOKEN|>"


class LlamaServerModel:
    """
    GGUF model accessed via a running llama-server (OpenAI-compatible completions API).

    model_spec format: "repo/ModelName-GGUF:Q4_K_M"
      e.g. "bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M"
    """

    def __init__(
        self,
        model_spec: str,
        base_url: str,
        request_model: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        self.model_spec = model_spec
        self.base_url = base_url.rstrip("/")
        self.request_model = request_model
        self.timeout = timeout
        self.max_retries = max_retries

    def _post_json(self, path: str, payload_data: dict) -> dict:
        payload = json.dumps(payload_data).encode()
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        def _request():
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())

        return call_with_retries(
            _request,
            f"llama-server request to {path}",
            retries=self.max_retries,
        )

    def _completions(self, prompt: str, max_tokens: int, stop: list[str]) -> dict:
        params = inference_params(max_tokens)
        params.pop("reasoning_effort", None)
        params.pop("chat_template_kwargs", None)
        params.pop("reasoning_format", None)
        payload_data = {
            "prompt": prompt,
            "stop": stop,
            **params,
        }
        if self.request_model:
            payload_data["model"] = self.request_model
        return self._post_json("/completions", payload_data)

    def _chat_completions(self, prompt: str, max_tokens: int | None) -> dict:
        params = chat_completion_params(max_tokens, model_name=self.model_spec)
        messages = []
        # Override Mistral's tool-focused default system prompt
        # (mode-collapses short-answer tasks at temp=0).
        if "mistral" in self.model_spec.lower():
            messages.append(
                {"role": "system", "content": "You are a helpful assistant."}
            )
        messages.append({"role": "user", "content": prompt})
        payload_data = {
            "messages": messages,
            **params,
        }
        if "aya-expanse" in self.model_spec.lower():
            payload_data["stop"] = [_AYA_END_OF_TURN_TOKEN]
        if self.request_model:
            payload_data["model"] = self.request_model
        return self._post_json("/chat/completions", payload_data)

    def _muse_completion(self, prompt: str, max_tokens: int | None) -> str:
        """Render Muse's direct-to-user template and bypass its reasoning channel."""
        from jinja2 import Template

        rendered = Template(_MUSE_USER_TEMPLATE.read_text(encoding="utf-8")).render(
            messages=[{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            bos_token=_MUSE_BOS_TOKEN,
        )
        data = self._completions(
            rendered,
            max_tokens=max_tokens or 256,
            stop=["<|eot|>", "<|eom|>", "<|start|>"],
        )
        return data["choices"][0]["text"].strip()

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        if "muse-glimmer" in self.model_spec.lower():
            return self._muse_completion(prompt, max_new_tokens)
        data = self._chat_completions(prompt, max_tokens=max_new_tokens)
        return data["choices"][0]["message"]["content"].strip()
