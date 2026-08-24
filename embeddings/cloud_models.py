"""Cloud embedding adapters exposing `.embed(texts, kind)` for model.py."""

from __future__ import annotations

import os
import time
from collections.abc import Sequence

import numpy as np


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms == 0, 1.0, norms)


GOOGLE_TASK_TYPES = {
    "query": "RETRIEVAL_QUERY",
    "doc": "RETRIEVAL_DOCUMENT",
    "symmetric": "SEMANTIC_SIMILARITY",
    "classification": "CLASSIFICATION",
}


def _env(*names: str) -> str:
    if value := next(
        (os.environ.get(name) for name in names if os.environ.get(name)), None
    ):
        return value
    joined = " / ".join(names)
    raise RuntimeError(f"{joined} env var is required")


def _retry(fn, attempts: int = 5):
    for attempt in range(attempts):
        try:
            return fn()
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2**attempt)


class CloudEmbedder:
    def __init__(self, provider: str, model: str, batch_size: int = 64):
        if provider not in {"openai", "google"}:
            raise ValueError(f"Unknown cloud provider: {provider!r}")
        self.provider = provider
        self.model = model
        self.batch_size = batch_size
        self._dim: int | None = None
        self.client, self.embed_config = self._make_client()

    def _make_client(self):
        if self.provider == "openai":
            from openai import OpenAI

            return OpenAI(api_key=_env("OPENAI_API_KEY")), None

        from google import genai
        from google.genai import types as gtypes

        return genai.Client(
            api_key=_env("GOOGLE_API_KEY", "GEMINI_API_KEY")
        ), gtypes.EmbedContentConfig

    def embed(self, texts: Sequence[str], kind: str = "symmetric") -> np.ndarray:
        if not texts:
            dim = self.get_embedding_dimension()
            return np.empty((0, dim), dtype=np.float32)

        vectors = [
            vector
            for i in range(0, len(texts), self.batch_size)
            for vector in _retry(
                lambda: self._embed_batch(texts[i : i + self.batch_size], kind)
            )
        ]
        arr = np.asarray(vectors, dtype=np.float32)
        self._dim = arr.shape[1]
        return _l2_normalize(arr)

    def _embed_batch(
        self, batch: Sequence[str], kind: str
    ) -> Sequence[Sequence[float]]:
        if self.provider == "openai":
            resp = self.client.embeddings.create(model=self.model, input=batch)
            return [d.embedding for d in resp.data]

        resp = self.client.models.embed_content(
            model=self.model,
            contents=batch,
            config=self.embed_config(
                task_type=GOOGLE_TASK_TYPES.get(kind, "SEMANTIC_SIMILARITY")
            ),
        )
        return [e.values for e in resp.embeddings]

    def get_embedding_dimension(self) -> int:
        if self._dim is None:
            self.embed(["x"])
        return self._dim  # type: ignore[return-value]


class OpenAIEmbedder(CloudEmbedder):
    def __init__(self, model: str, batch_size: int = 64):
        super().__init__("openai", model, batch_size)


class GoogleEmbedder(CloudEmbedder):
    def __init__(self, model: str, batch_size: int = 64):
        super().__init__("google", model, batch_size)


def load_cloud_model(provider: str, model_id: str, batch_size: int):
    return CloudEmbedder(provider, model_id, batch_size)
