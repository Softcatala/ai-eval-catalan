import json
import time
import urllib.request
from pathlib import Path

from inference import chat_completion_params, inference_params


def _is_gguf_model(model_name: str) -> bool:
    """Return True if model_name is a GGUF spec (repo:quantization or .gguf file)."""
    return "GGUF" in model_name or "gguf" in model_name or model_name.endswith(".gguf")


_FILENAME_OVERRIDES = {
    "RichardErkhov/BSC-LT_-_salamandra-7b-instruct-gguf": "salamandra-7b-instruct.{quant}.gguf",
    "mradermacher/salamandra-7b-instruct-2606-GGUF": "salamandra-7b-instruct-2606.{quant}.gguf",
}


def expected_gguf_filename(model_spec: str) -> str:
    """Return the GGUF filename expected for a repo:quant spec or local .gguf path."""
    if model_spec.lower().endswith(".gguf"):
        return Path(model_spec).name

    if ":" in model_spec:
        repo, quant = model_spec.rsplit(":", 1)
    else:
        repo, quant = model_spec, "Q8_0"

    if repo in _FILENAME_OVERRIDES:
        return _FILENAME_OVERRIDES[repo].format(quant=quant)

    model_base = repo.split("/")[-1].replace("-GGUF", "")
    return f"{model_base}-{quant}.gguf"


def _hf_tokenizer_from_gguf(model_spec: str) -> str:
    """
    Derive the HuggingFace tokenizer repo from a bartowski GGUF spec.
    e.g. "bartowski/google_gemma-3-1b-it-GGUF:Q8_0" -> "google/gemma-3-1b-it"
    """
    _KNOWN = {
        "aya-expanse-8b": "CohereForAI/aya-expanse-8b",
        "EuroLLM-9B-Instruct": "utter-project/EuroLLM-9B-Instruct",
        "BSC-LT_-_salamandra-7b-instruct-gguf": "BSC-LT/salamandra-7b-instruct",
        "salamandra-7b-instruct-2606": "BSC-LT/salamandra-7b-instruct-2606",
        "gemma-4-12b-it": "google/gemma-4-12b-it",
    }

    repo = model_spec.rsplit(":", 1)[0]
    name = repo.split("/")[-1]
    name = name.replace("-GGUF", "")

    if name in _KNOWN:
        return _KNOWN[name]

    if "_" in name:
        org, model = name.split("_", 1)
        return f"{org}/{model}"
    if name.startswith("Llama") or name.startswith("Meta-Llama"):
        return f"meta-llama/{name}"
    return name


class LlamaServerModel:
    """
    GGUF model accessed via a running llama-server (OpenAI-compatible completions API).

    model_spec format: "repo/ModelName-GGUF:Q8_0"
      e.g. "bartowski/Llama-3.2-3B-Instruct-GGUF:Q8_0"
    """

    def __init__(
        self,
        model_spec: str,
        base_url: str,
        request_model: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 1,
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
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read())
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    print(f"[warn] llama-server request failed, retrying: {e}", flush=True)
                    time.sleep(1)
        raise last_error

    def _completions(self, prompt: str, max_tokens: int, **kwargs) -> dict:
        params = inference_params(max_tokens)
        params.pop("reasoning_effort", None)
        params.pop("chat_template_kwargs", None)
        params.pop("reasoning_format", None)
        payload_data = {
            "prompt": prompt,
            **params,
            **kwargs,
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
            messages.append({"role": "system", "content": "You are a helpful assistant."})
        messages.append({"role": "user", "content": prompt})
        payload_data = {
            "messages": messages,
            **params,
        }
        if self.request_model:
            payload_data["model"] = self.request_model
        return self._post_json("/chat/completions", payload_data)

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        try:
            data = self._chat_completions(prompt, max_tokens=max_new_tokens)
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[warn] llama-server generation failed: {e}", flush=True)
            return ""

    def score_options(self, prompt: str, options: list[str]) -> int:
        """Pick the option with the highest token log-probability sum."""
        scores = []
        for opt in options:
            try:
                data = self._completions(prompt + opt, max_tokens=1, echo=True, logprobs=1)
                token_logprobs = data["choices"][0]["logprobs"]["token_logprobs"]
                valid = [lp for lp in token_logprobs if lp is not None]
                scores.append(sum(valid))
            except Exception as e:
                print(f"[warn] llama-server option scoring failed: {e}", flush=True)
                scores.append(float("-inf"))
        return scores.index(max(scores))
