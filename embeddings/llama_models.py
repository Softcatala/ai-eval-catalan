"""Local llama.cpp embedding adapter for the OpenAI-compatible endpoint."""

import json
import urllib.request

import numpy as np


class LlamaEmbedder:
    def __init__(self, server_url, model, batch_size):
        self.url = server_url.rstrip("/")
        self.model = model
        self.batch_size = batch_size
        self.dim = None
        self.max_seq_length = None

    def request(self, url, payload):
        request = urllib.request.Request(
            url,
            data=json.dumps({"model": self.model, **payload}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.load(response)

    def embed(self, texts, kind="symmetric"):
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            if self.max_seq_length is not None:
                batch = [
                    self.request(
                        self.url.removesuffix("/v1") + "/tokenize",
                        {"content": text, "add_special": True},
                    )["tokens"][: self.max_seq_length]
                    for text in batch
                ]
            data = sorted(
                self.request(self.url + "/embeddings", {"input": batch})["data"],
                key=lambda row: row["index"],
            )
            if [row["index"] for row in data] != list(range(len(batch))):
                raise ValueError("Embedding response does not match the input batch")
            vectors.extend(row["embedding"] for row in data)
        embeddings = np.asarray(vectors, dtype=np.float32)
        if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
            raise ValueError("Invalid embedding vectors returned by llama.cpp")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if (norms == 0).any():
            raise ValueError("Zero embedding vector returned by llama.cpp")
        self.dim = embeddings.shape[1]
        return embeddings / norms

    def get_embedding_dimension(self):
        if self.dim is None:
            self.embed(["x"])
        return self.dim
