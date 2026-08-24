"""
Evaluate a sentence-transformer embedding model on Catalan tasks:
  - STS-ca semantic textual similarity, Spearman correlation of cosine vs gold.
  - XQuAD-ca retrieval (question -> context), harder retrieval (low lexical overlap).
  - TeCla topic classification (4 coarse labels) via linear probe.
  - 95% bootstrap CIs on nDCG@10 and Spearman.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer

RNG = np.random.default_rng(42)
BOOTSTRAP_ITERS = 1000


def encode(model, texts, batch_size, kind="symmetric"):
    if hasattr(model, "embed"):
        return model.embed(texts, kind=kind)
    return model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=batch_size,
    )


def ndcg10_from_ranks(ranks):
    ranks = np.asarray(ranks)
    return float(np.mean(np.where(ranks < 10, 1 / np.log2(ranks + 2), 0.0)))


def retrieval_metrics(sims, gold_idx):
    """Return retrieval metrics plus ranks; sims[i, j] scores query i vs doc j."""
    gold_idx = np.asarray(gold_idx)
    gold_scores = sims[np.arange(len(gold_idx)), gold_idx][:, None]
    ranks = np.sum(sims > gold_scores, axis=1).astype(int)
    return {
        "ndcg_at_10": round(ndcg10_from_ranks(ranks), 4),
    }, ranks


def bootstrap_ci(values, statistic, n_iter=BOOTSTRAP_ITERS):
    n = len(values)
    stats = [statistic(values[RNG.integers(0, n, size=n)]) for _ in range(n_iter)]
    return round(float(np.percentile(stats, 2.5)), 4), round(
        float(np.percentile(stats, 97.5)), 4
    )


def prefixed(records, field, prefix):
    return [prefix + r[field] for r in records]


def encode_pair(model, left, right, batch_size, left_kind="query", right_kind="doc"):
    return (
        encode(model, left, batch_size, kind=left_kind),
        encode(model, right, batch_size, kind=right_kind),
    )


def retrieval_from_embeddings(query_emb, corpus_emb, gold):
    metrics, ranks = retrieval_metrics(query_emb @ corpus_emb.T, gold)
    metrics["ndcg_at_10_ci"] = list(bootstrap_ci(ranks, ndcg10_from_ranks))
    return metrics


def eval_retrieval(
    model, queries, corpus, gold, batch_size, query_kind="query", doc_kind="doc"
):
    return retrieval_from_embeddings(
        *encode_pair(model, queries, corpus, batch_size, query_kind, doc_kind),
        gold,
    )


def eval_xquad_ca(model, query_prefix, doc_prefix, batch_size):
    ds = load_dataset("projecte-aina/xquad-ca", split="test")
    questions = prefixed(ds, "question", query_prefix)
    ctx_to_idx = {}
    gold = [ctx_to_idx.setdefault(r["context"], len(ctx_to_idx)) for r in ds]
    corpus = [doc_prefix + context for context in ctx_to_idx]

    return eval_retrieval(model, questions, corpus, gold, batch_size) | {
        "n_queries": len(questions),
        "n_corpus": len(corpus),
    }


def eval_sts_ca(model, sym_prefix, batch_size):
    ds = load_dataset("projecte-aina/sts-ca", split="test")
    s1 = prefixed(ds, "sentence_1", sym_prefix)
    s2 = prefixed(ds, "sentence_2", sym_prefix)
    gold = np.asarray([r["label"] for r in ds], dtype=np.float32)

    e1, e2 = encode_pair(model, s1, s2, batch_size, "symmetric", "symmetric")
    cos = np.sum(e1 * e2, axis=1)

    lo, hi = bootstrap_ci(
        np.column_stack([cos, gold]),
        lambda P: float(spearmanr(P[:, 0], P[:, 1]).statistic),
    )
    return {
        "spearman": round(float(spearmanr(cos, gold).statistic), 4),
        "spearman_ci": [lo, hi],
        "n_pairs": len(s1),
    }


def eval_tecla(model, doc_prefix, batch_size, n_train):
    """Linear probe on TeCla label1 (4 coarse classes: Societat/Política/Economia/Cultura)."""
    train = (
        load_dataset("projecte-aina/tecla", split="train")
        .shuffle(seed=42)
        .select(range(n_train))
    )
    test = load_dataset("projecte-aina/tecla", split="test")

    X_train = encode(
        model,
        prefixed(train, "sentence", doc_prefix),
        batch_size,
        kind="classification",
    )
    y_train = [r["label1"] for r in train]
    X_test = encode(
        model, prefixed(test, "sentence", doc_prefix), batch_size, kind="classification"
    )
    y_test = [r["label1"] for r in test]

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    return {
        "macro_f1": round(float(f1_score(y_test, pred, average="macro")), 4),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_classes": len(set(y_train)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--display-name", default=None)
    p.add_argument("--n-tecla-train", type=int, default=10000)
    p.add_argument("--query-prefix", default="")
    p.add_argument("--doc-prefix", default="")
    p.add_argument("--sts-prefix", default=None)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-seq-length", type=int, default=None)
    p.add_argument("--skip", default="", help="Comma-separated benchmark names to skip")
    p.add_argument(
        "--cloud-provider",
        choices=["openai", "google"],
        default=None,
        help="When set, --model is the cloud model id (e.g. text-embedding-3-large); keys read from env",
    )
    a = p.parse_args()

    skip = {s.strip() for s in a.skip.split(",") if s.strip()}
    if a.cloud_provider:
        from cloud_models import load_cloud_model

        model = load_cloud_model(a.cloud_provider, a.model, batch_size=a.batch_size)
    else:
        model = SentenceTransformer(
            a.model, trust_remote_code=a.trust_remote_code, device=a.device
        )
    if a.max_seq_length is not None:
        if not hasattr(model, "max_seq_length"):
            raise ValueError(
                "--max-seq-length is only supported for local SentenceTransformer models"
            )
        model.max_seq_length = a.max_seq_length

    sts_prefix = a.sts_prefix if a.sts_prefix is not None else a.query_prefix
    tasks = [
        (
            "xquad_ca_retrieval",
            "xquad_ca",
            lambda: eval_xquad_ca(model, a.query_prefix, a.doc_prefix, a.batch_size),
        ),
        ("sts_ca", "sts_ca", lambda: eval_sts_ca(model, sts_prefix, a.batch_size)),
        (
            "tecla_classification",
            "tecla",
            lambda: eval_tecla(model, a.doc_prefix, a.batch_size, a.n_tecla_train),
        ),
    ]

    benchmarks = {
        name: run() for name, skip_name, run in tasks if skip_name not in skip
    }

    result = {
        "model": a.model,
        "display_name": a.display_name or a.model.split("/")[-1],
        "cloud": bool(a.cloud_provider),
        "embedding_dim": (
            model.get_embedding_dimension()
            if hasattr(model, "get_embedding_dimension")
            else model.get_sentence_embedding_dimension()
        ),
        "benchmarks": benchmarks,
    }
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(result, indent=2, ensure_ascii=False))

    xq = benchmarks.get("xquad_ca_retrieval", {})
    sts = benchmarks.get("sts_ca", {})
    tec = benchmarks.get("tecla_classification", {})
    print(
        f">>> {result['display_name']}: "
        f"XQuAD nDCG@10={xq.get('ndcg_at_10')} | "
        f"STS-ca={sts.get('spearman')} | "
        f"TeCla F1={tec.get('macro_f1')}"
    )


if __name__ == "__main__":
    main()
