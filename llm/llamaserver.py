import json
import os
import subprocess
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path


def _is_gguf_model(model_name: str) -> bool:
    """Return True if model_name is a GGUF spec (repo:quantization or .gguf file)."""
    return "GGUF" in model_name or "gguf" in model_name or model_name.endswith(".gguf")


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


def _is_thinking_model(model_spec: str) -> bool:
    """Return True for models known to emit thinking tokens (e.g. Gemma-4 E4B, Qwen3/3.5)."""
    lower = model_spec.lower()
    return (
        "gemma-4" in lower
        or "gemma4" in lower
        or "-e4b" in lower
        or "qwen3" in lower
    )


def _wait_for_port(port: int, timeout: float = 300.0):
    """Block until llama-server is ready (model loaded) or timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=2
            ) as resp:
                data = json.loads(resp.read())
                if data.get("status") == "ok":
                    return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(
        f"llama-server did not become ready on port {port} within {timeout}s"
    )


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
        payload_data = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            **kwargs,
        }
        if self.request_model:
            payload_data["model"] = self.request_model
        return self._post_json("/completions", payload_data)

    def _chat_completions(self, prompt: str, max_tokens: int) -> dict:
        payload_data = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        if self.request_model:
            payload_data["model"] = self.request_model
        return self._post_json("/chat/completions", payload_data)

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
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


@contextmanager
def llama_server_context(model_spec: str, port: int, device: str = "cpu", extra_args: list | None = None):
    """
    Download the GGUF file via huggingface_hub, spawn llama-server, and yield the base_url.
    """
    from huggingface_hub import get_token, hf_hub_url

    if ":" in model_spec:
        repo, quant = model_spec.rsplit(":", 1)
    else:
        repo, quant = model_spec, "Q8_0"

    _FILENAME_OVERRIDES = {
        "RichardErkhov/BSC-LT_-_salamandra-7b-instruct-gguf": "salamandra-7b-instruct.{quant}.gguf",
        "mradermacher/salamandra-7b-instruct-2606-GGUF": "salamandra-7b-instruct-2606.{quant}.gguf",
    }
    if repo in _FILENAME_OVERRIDES:
        filename = _FILENAME_OVERRIDES[repo].format(quant=quant)
    else:
        model_base = repo.split("/")[-1].replace("-GGUF", "")
        filename = f"{model_base}-{quant}.gguf"

    print(f"[server] Ensuring {filename} is cached locally …", flush=True)
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    blob_dir = os.path.join(cache_dir, f"models--{repo.replace('/', '--')}", "blobs")
    incomplete = os.path.join(blob_dir, filename + ".incomplete")
    blob_path = os.path.join(blob_dir, filename)
    if os.path.exists(blob_path):
        local_path = blob_path
        print(f"[server] Already cached at {local_path}", flush=True)
    else:
        url = hf_hub_url(repo_id=repo, filename=filename)
        token = get_token()
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1 MB
            os.makedirs(blob_dir, exist_ok=True)
            with open(incomplete, "ab") as f:
                downloaded = f.seek(0, 2)  # resume if partial
            if downloaded > 0 and total > 0:
                req2 = urllib.request.Request(
                    url, headers={**headers, "Range": f"bytes={downloaded}-"}
                )
                try:
                    resp2 = urllib.request.urlopen(req2)
                except Exception:
                    resp2 = urllib.request.urlopen(req)
                    downloaded = 0
            else:
                resp2 = resp
            with open(incomplete, "ab") as f:
                while True:
                    chunk = resp2.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        mb = downloaded / 1024 / 1024
                        total_mb = total / 1024 / 1024
                        print(
                            f"\r[server] Downloading {filename}: {mb:.0f}/{total_mb:.0f} MB ({pct:.1f}%)",
                            end="",
                            flush=True,
                        )
            print(flush=True)
        os.rename(incomplete, blob_path)
        local_path = blob_path

    log_path = Path(f"llama_server_{port}.log")
    print(
        f"[server] Starting llama-server on port {port} (device={device}) … (log: {log_path})"
    )
    default_server = Path(__file__).parent.parent.parent / "llama.cpp" / "build" / "bin" / "llama-server"
    llama_server_bin = os.environ.get("LLAMA_SERVER_PATH", str(default_server))
    cmd = [
        llama_server_bin,
        "--model",
        local_path,
        "--port",
        str(port),
        "--ctx-size",
        "2048",
        "--parallel",
        "1",
    ]
    if device == "cuda":
        cmd += ["--n-gpu-layers", "99"]
    if extra_args:
        cmd += extra_args
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=log_file,
        env=os.environ.copy(),
    )
    try:
        _wait_for_port(port)
        print(f"[server] Ready at http://127.0.0.1:{port}")
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        print("[server] Stopping llama-server …")
        proc.terminate()
        proc.wait()
        log_file.close()
