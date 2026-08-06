"""Summarize embedding eval results into a console table and embeddings.json."""

import argparse
import json
from pathlib import Path


CLOUD_PREFIXES = ("google-", "openai-", "gemini-", "voyage-", "cohere-")
CLOUD_MODEL_REPOS = {
    "gemini-embedding-001": "https://ai.google.dev/gemini-api/docs/embeddings",
    "text-embedding-3-large": "https://platform.openai.com/docs/models/text-embedding-3-large",
    "text-embedding-3-small": "https://platform.openai.com/docs/models/text-embedding-3-small",
}


def is_cloud(d) -> bool:
    if "cloud" in d and d["cloud"] is not None:
        return bool(d["cloud"])
    name = (d.get("display_name") or d.get("model") or "").lower()
    return any(name.startswith(p) for p in CLOUD_PREFIXES)


def repo_url(model: str) -> str:
    if "/" in model:
        return f"https://huggingface.co/{model}"
    if model in CLOUD_MODEL_REPOS:
        return CLOUD_MODEL_REPOS[model]
    raise ValueError(f"No repo URL configured for model: {model}")


def load_rows() -> list[dict]:
    rows = []
    for p in sorted(Path("evals").glob("results_*.json")):
        d = json.loads(p.read_text())
        bench = d.get("benchmarks", {})
        xq = bench.get("xquad_ca_retrieval", {})
        sts = bench.get("sts_ca", {})
        tec = bench.get("tecla_classification", {})
        rows.append({
            "model": d["display_name"],
            "repo_url": repo_url(d["model"]),
            "cloud": is_cloud(d),
            "dim": d["embedding_dim"],
            "xquad_ndcg_at_10": xq.get("ndcg_at_10"),
            "xquad_ndcg_at_10_ci": xq.get("ndcg_at_10_ci"),
            "sts_ca_spearman": sts.get("spearman"),
            "sts_ca_spearman_ci": sts.get("spearman_ci"),
            "tecla_macro_f1": tec.get("macro_f1"),
        })
    return rows


# Composite: plain mean of the three primary task scores (all bounded in [0, 1]).
# XQuAD nDCG@10, STS-ca Spearman, TeCla macro F1.
def composite(r) -> float:
    vs = [r["xquad_ndcg_at_10"], r["sts_ca_spearman"], r["tecla_macro_f1"]]
    vs = [v for v in vs if v is not None]
    return sum(vs) / len(vs) if vs else -1


COL_LABELS = {
    "model": "Model",
    "cloud": "Cloud",
    "dim": "Dim",
    "xquad_ndcg_at_10": "XQuAD nDCG@10",
    "sts_ca_spearman": "STS-ca Sp",
    "tecla_macro_f1": "TeCla F1",
    "composite": "Puntuació composta",
}


def main():
    parser = argparse.ArgumentParser(description="Summarize embeddings eval results")
    parser.add_argument("--json-out", default="embeddings.json")
    args = parser.parse_args()

    rows = load_rows()
    for r in rows:
        r["composite"] = round(composite(r), 4)
    rows.sort(key=lambda r: r["composite"], reverse=True)

    w = max(len(r["model"]) for r in rows) + 2
    cols = [
        ("Dim", "dim", 6, "d"),
        ("XQuAD nDCG", "xquad_ndcg_at_10", 12, ".4f"),
        ("STS-ca Sp", "sts_ca_spearman", 11, ".4f"),
        ("TeCla F1", "tecla_macro_f1", 10, ".4f"),
        ("Avg", "composite", 8, ".4f"),
    ]
    header = f"{'Model':<{w}}" + "".join(f"{c[0]:>{c[2]}}" for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        line = f"{r['model']:<{w}}"
        for label, key, width, fmt in cols:
            v = r.get(key)
            line += f"{'-':>{width}}" if v is None else f"{v:>{width}{fmt}}"
        print(line)

    # Gaps under ~0.015 on the composite are within sampling noise (STS-ca: 500 pairs,
    # XQuAD-ca: ~1.2k queries, TeCla: 17k test items). Treat clustered middle ranks as tied.
    # Per-model 95% bootstrap CIs are stored in embeddings.json (xquad_ndcg_at_10_ci,
    # sts_ca_spearman_ci) for anyone who wants the exact intervals.
    print()
    print("Note: composite gaps under ~0.015 are within sampling noise — treat clustered "
          "middle ranks as tied. Per-model 95% bootstrap CIs are in embeddings.json.")

    out = {
        "text": COL_LABELS,
        "data": [{**r, "model": f"(*) {r['model']}"} if r["cloud"] else r for r in rows],
    }
    Path(args.json_out).write_text(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
