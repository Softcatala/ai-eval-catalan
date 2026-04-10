"""
Catalan Linguistic Competency Evaluation Pipeline
Evaluates GGUF models on 4 key Catalan benchmarks:
  1. CatCoLA  – grammatical acceptability
  2. CLUB     – NER, STS, QA (core NLP tasks)
  3. IberBench – broad NLP via lm-evaluation-harness
  4. FLORES+  – machine translation quality

Requirements:
  pip install datasets scikit-learn sacrebleu lm_eval huggingface_hub
  llama-server must be available in PATH (from llama.cpp)

Usage:
  # With a llama.cpp GGUF model from bartowski (default):
  python model.py --model "bartowski/Llama-3.2-3B-Instruct-GGUF:Q8_0"

  # With the Google AI API (Gemma 4):
  python model.py --model "gemini" --api-key "YOUR_KEY"

  # Run only specific benchmarks:
  python model.py --model "bartowski/Llama-3.2-3B-Instruct-GGUF:Q8_0" --benchmarks catcola flores
"""

import argparse
import gc
import json
import math
import os
import re
import subprocess
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

# /mnt/sda1 may not be mounted — strip broken HF cache paths so datasets
# falls back to the default ~/.cache/huggingface location.
for _key in ("HF_HOME", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE"):
    if _key in os.environ and not os.path.isdir(os.environ[_key]):
        del os.environ[_key]

# facebook/flores uses an old dataset script — trust it so datasets doesn't refuse to load it.
os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")

from datasets import load_dataset
from sklearn.metrics import matthews_corrcoef, accuracy_score

# ── Optional: lm_eval for IberBench tasks ─────────────────────────────────────
try:
    import lm_eval
    import lm_eval.models.openai_completions as _lm_oai

    def _patched_parse_logprobs(self, outputs, tokens=None, ctxlens=None, **kwargs):
        """
        Patch lm_eval's parse_logprobs to handle llama-server's v2 logprobs format
        (logprobs.content[].logprob) in addition to the v1 format (token_logprobs[]).
        """
        res = []
        for out, toks, ctxlen in zip(outputs, tokens, ctxlens):
            choice = out["choices"][0]
            logprobs_data = choice.get("logprobs", {}) or {}

            if "token_logprobs" in logprobs_data:
                # v1 format
                lps = logprobs_data["token_logprobs"][ctxlen:-1]
            elif "content" in logprobs_data:
                # v2 format returned by newer llama-server
                lps = [c["logprob"] for c in logprobs_data["content"]]
                lps = lps[ctxlen - 1 : -1] if ctxlen > 0 else lps[:-1]
            else:
                lps = []

            continuation_logprob = sum(lps)
            # greedy: check if each predicted token matches the top logprob token
            is_greedy = True
            content = logprobs_data.get("content", [])
            for i, c in enumerate(
                content[ctxlen - 1 : -1] if ctxlen > 0 else content[:-1]
            ):
                top = (c.get("top_logprobs") or [{}])[0]
                if top.get("token") != c.get("token"):
                    is_greedy = False
                    break
            res.append((continuation_logprob, is_greedy))
        return res

    _lm_oai.LocalCompletionsAPI.parse_logprobs = _patched_parse_logprobs
    HAS_LM_EVAL = True
except ImportError:
    HAS_LM_EVAL = False
    print("[warn] lm_eval not found – IberBench tasks will be skipped.")
    print("       Install with: pip install lm_eval")


# ──────────────────────────────────────────────────────────────────────────────
# Model wrapper — supports llama-server (GGUF) and Google AI API
# ──────────────────────────────────────────────────────────────────────────────


def _is_gguf_model(model_name: str) -> bool:
    """Return True if model_name is a GGUF spec (repo:quantization or .gguf file)."""
    return "GGUF" in model_name or model_name.endswith(".gguf")


def _hf_tokenizer_from_gguf(model_spec: str) -> str:
    """
    Derive the HuggingFace tokenizer repo from a bartowski GGUF spec.
    e.g. "bartowski/google_gemma-3-1b-it-GGUF:Q8_0" -> "google/gemma-3-1b-it"
    """
    # Models whose names don't encode the HF org — map them explicitly.
    _KNOWN = {
        "aya-expanse-8b": "CohereForAI/aya-expanse-8b",
        "EuroLLM-9B-Instruct": "utter-project/EuroLLM-9B-Instruct",
    }

    repo = model_spec.rsplit(":", 1)[0]  # strip :Q8_0
    name = repo.split("/")[-1]  # google_gemma-3-1b-it-GGUF
    name = name.replace("-GGUF", "")  # google_gemma-3-1b-it

    if name in _KNOWN:
        return _KNOWN[name]

    # bartowski prefixes the original org with an underscore
    if "_" in name:
        org, model = name.split("_", 1)
        return f"{org}/{model}"
    # No org prefix — use well-known defaults
    if name.startswith("Llama") or name.startswith("Meta-Llama"):
        return f"meta-llama/{name}"
    return name


class LlamaServerModel:
    """
    GGUF model accessed via a running llama-server (OpenAI-compatible completions API).

    model_spec format: "repo/ModelName-GGUF:Q8_0"
      e.g. "bartowski/Llama-3.2-3B-Instruct-GGUF:Q8_0"
    """

    def __init__(self, model_spec: str, base_url: str):
        self.model_spec = model_spec
        self.base_url = base_url.rstrip("/")

    def _completions(self, prompt: str, max_tokens: int, **kwargs) -> dict:
        payload = json.dumps(
            {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0,
                **kwargs,
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.base_url}/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())

    def _chat_completions(self, prompt: str, max_tokens: int) -> dict:
        payload = json.dumps(
            {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0,
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        data = self._chat_completions(prompt, max_tokens=max_new_tokens)
        return data["choices"][0]["message"]["content"].strip()

    def score_options(self, prompt: str, options: list[str]) -> int:
        """Pick the option with the highest token log-probability sum."""
        scores = []
        for opt in options:
            data = self._completions(prompt + opt, max_tokens=1, echo=True, logprobs=1)
            token_logprobs = data["choices"][0]["logprobs"]["token_logprobs"]
            valid = [lp for lp in token_logprobs if lp is not None]
            scores.append(sum(valid))
        return scores.index(max(scores))


class GeminiModel:
    """Google AI API wrapper (for Gemma 4 / Gemini models)."""

    def __init__(self, api_key: str, model_name: str = "gemma-3-27b-it"):
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        # Thinking models (gemini-3.x) consume tokens for internal reasoning before
        # producing output — a small budget leaves nothing for the actual response.
        effective_tokens = max(max_new_tokens, 1024)
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": effective_tokens,
                    "temperature": 0,
                },
            )
            candidate = response.candidates[0] if response.candidates else None
            if candidate is None:
                return ""
            # finish_reason: 1=STOP (normal), 2=SAFETY, 3=RECITATION, etc.
            # Even with finish_reason=1 the response can have no parts (empty output).
            parts = getattr(candidate.content, "parts", None) if candidate.content else None
            if not parts:
                return ""
            text = "".join(p.text for p in parts if hasattr(p, "text"))
            return text.strip()
        except Exception as e:
            print(f"[error] API call failed: {e}")
            time.sleep(2)
            return ""

    def score_options(self, prompt: str, options: list[str]) -> int:
        """For API models, generate and pick the closest option."""
        answer = self.generate(prompt + "\nAnswer with only A, B, C, or D.")
        for i, opt in enumerate(options):
            if opt.strip().lower() in answer.lower():
                return i
        return 0  # fallback


class OpenAIModel:
    """OpenAI-compatible API wrapper (works with OpenAI and OpenRouter)."""

    def __init__(self, api_key: str, model_name: str, base_url: str | None = None):
        from openai import OpenAI

        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        effective_tokens = max(max_new_tokens, 1024)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=effective_tokens,
                temperature=0,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"[error] API call failed: {e}")
            time.sleep(2)
            return ""

    def score_options(self, prompt: str, options: list[str]) -> int:
        answer = self.generate(prompt + "\nAnswer with only A, B, C, or D.")
        for i, opt in enumerate(options):
            if opt.strip().lower() in answer.lower():
                return i
        return 0


# ──────────────────────────────────────────────────────────────────────────────
# 1. VeritasQA — Generative / Open-ended QA in Catalan
# ──────────────────────────────────────────────────────────────────────────────


def run_veritasqa(model, n_samples: int = 100) -> dict:
    """
    Open-ended truthfulness QA in Catalan.
    The model generates a free-form answer; it is scored as correct if it
    matches any of the accepted correct answers (case-insensitive substring).
    Metric: Accuracy.
    Dataset: projecte-aina/veritasQA (test split, Catalan rows).
    """
    print("\n[1/7] Running VeritasQA (generative QA) …")
    ds = load_dataset("projecte-aina/veritasQA", "ca", split="test")
    ca_rows = list(ds)
    limit = min(n_samples, len(ca_rows))

    correct = 0
    for i in range(limit):
        item = ca_rows[i]
        question = item["question"]
        correct_answers = item.get("correct_answers") or [item.get("best_answer", "")]

        prompt = (
            f"Respon la següent pregunta en català amb una resposta breu.\n\n"
            f"Pregunta: {question}\nResposta:"
        )
        pred = model.generate(prompt, max_new_tokens=64).strip().lower()
        if any(ans.strip().lower() in pred for ans in correct_answers if ans):
            correct += 1

    del ds
    gc.collect()

    acc = round(correct / limit, 4) if limit else 0.0
    result = {"accuracy": acc, "n": limit}
    print(f"    ✓ Accuracy={acc:.4f}  (n={limit})")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 2. STS-ca — Semantic Textual Similarity (Paraphrase proxy)
# ──────────────────────────────────────────────────────────────────────────────


def run_sts_ca(model, n_samples: int = 100) -> dict:
    """
    Semantic textual similarity as a proxy for paraphrase quality.
    The model is given two sentences and asked to rate their similarity (0–5).
    Metric: Pearson correlation between predicted and gold scores.
    Dataset: projecte-aina/sts-ca (test split).
    """
    print("\n[2/7] Running STS-ca (paraphrase / similarity) …")
    ds = load_dataset("projecte-aina/sts-ca", split="test")
    limit = min(n_samples, len(ds))

    gold, pred_scores = [], []
    for i in range(limit):
        item = ds[i]
        s1, s2, label = item["sentence_1"], item["sentence_2"], item["label"]

        prompt = (
            "Puntua la similitud semàntica entre les dues frases següents amb un número de 0 a 5, "
            "on 0 significa cap similitud i 5 significa idèntic significat. "
            "Respon només amb el número.\n\n"
            f"Frase 1: {s1}\nFrase 2: {s2}\nPuntuació:"
        )
        raw = model.generate(prompt, max_new_tokens=16).strip()
        # Extract first number found in the response
        m = re.search(r"[0-5](?:\.[0-9]+)?", raw)
        score = float(m.group()) if m else 2.5  # fallback to midpoint

        gold.append(float(label))
        pred_scores.append(score)

    del ds
    gc.collect()

    # Pearson correlation
    n = len(gold)
    mean_g = sum(gold) / n
    mean_p = sum(pred_scores) / n
    cov = sum((g - mean_g) * (p - mean_p) for g, p in zip(gold, pred_scores)) / n
    std_g = math.sqrt(sum((g - mean_g) ** 2 for g in gold) / n)
    std_p = math.sqrt(sum((p - mean_p) ** 2 for p in pred_scores) / n)
    pearson = round(cov / (std_g * std_p), 4) if std_g > 0 and std_p > 0 else 0.0

    result = {"pearson": pearson, "n": n}
    print(f"    ✓ Pearson={pearson:.4f}  (n={n})")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 3. CatCoLA — Catalan Corpus of Linguistic Acceptability
# ──────────────────────────────────────────────────────────────────────────────


def run_catcola(model, n_samples: int = 200) -> dict:
    """
    Binary classification: is a Catalan sentence grammatically acceptable?
    Metric: Matthews Correlation Coefficient (MCC), as per CoLA standard.
    Dataset: nbel/CatCoLA on HuggingFace.
    """
    print("\n[3/7] Running CatCoLA …")
    ds = load_dataset("nbel/CatCoLA", split="validation")
    limit = min(n_samples, len(ds))

    preds, labels = [], []
    for i in range(limit):
        item = ds[i]
        sentence = item["Sentence"]
        label = item["Label"]  # 0 = unacceptable, 1 = acceptable

        prompt = (
            "La seguent frase en catala es gramaticalment correcta? "
            "Respon nomes amb 'si' o 'no'.\n\n"
            f"Frase: {sentence}\nResposta:"
        )
        answer = model.generate(prompt, max_new_tokens=16).lower()
        pred = 1 if "si" in answer or "sí" in answer else 0

        preds.append(pred)
        labels.append(label)

    del ds
    gc.collect()

    mcc = matthews_corrcoef(labels, preds)
    acc = accuracy_score(labels, preds)
    result = {"mcc": round(mcc, 4), "accuracy": round(acc, 4), "n": len(preds)}
    print(f"    ✓ MCC={mcc:.4f}  Accuracy={acc:.4f}  (n={len(preds)})")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 4. CLUB — Catalan Language Understanding Benchmark (QA slice)
# ──────────────────────────────────────────────────────────────────────────────


def run_club_qa(model, n_samples: int = 100) -> dict:
    """
    Extractive QA on VilaQuAD (Catalan Wikipedia QA).
    Metric: Exact Match (EM) and token-level F1.
    Dataset: projecte-aina/vilaquad on HuggingFace.
    """
    print("\n[4/7] Running CLUB / VilaQuAD (QA) …")
    ds = load_dataset("projecte-aina/vilaquad", split="validation")

    def _iter_qa_pairs(dataset, limit):
        """Yield QA pairs lazily from the nested vilaquad structure."""
        count = 0
        for item in dataset:
            for para in item["data"]["paragraphs"]:
                context = para["context"]
                for qa in para["qas"]:
                    yield context, qa["question"], [a["text"] for a in qa["answers"]]
                    count += 1
                    if count >= limit:
                        return

    exact_matches = []
    for context, question, gold_answers in _iter_qa_pairs(ds, n_samples):
        prompt = (
            f"Llegeix el text i respon la pregunta amb una frase curta extreta del text.\n\n"
            f"Text: {context[:800]}\n\nPregunta: {question}\nResposta:"
        )
        pred = model.generate(prompt, max_new_tokens=64).strip().lower()
        em = any(gold.strip().lower() in pred for gold in gold_answers)
        exact_matches.append(int(em))

    del ds
    gc.collect()

    score = sum(exact_matches) / len(exact_matches)
    result = {"exact_match_approx": round(score, 4), "n": len(exact_matches)}
    print(f"    ✓ Approx. Exact Match={score:.4f}  (n={len(exact_matches)})")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 3. CaSum — Catalan Summarization
# ──────────────────────────────────────────────────────────────────────────────


def run_casum(model, n_samples: int = 100) -> dict:
    """
    Abstractive summarization on CaSum (Catalan news articles → headline).
    Metric: ROUGE-1, ROUGE-2, ROUGE-L F1.
    Dataset: projecte-aina/casum on HuggingFace (test split).
    """
    from rouge_score import rouge_scorer

    print("\n[5/7] Running CaSum (summarization) …")
    ds = load_dataset("projecte-aina/casum", split="test")
    limit = min(n_samples, len(ds))
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for i in range(limit):
        item = ds[i]
        article = item["text"][:1200]  # truncate to fit context
        reference = item["summary"]

        prompt = (
            "Llegeix el següent article en català i escriu un titular breu que el resumeixi.\n\n"
            f"Article: {article}\n\nTitular:"
        )
        pred = model.generate(prompt, max_new_tokens=64).strip()
        s = scorer.score(reference, pred)
        scores["rouge1"].append(s["rouge1"].fmeasure)
        scores["rouge2"].append(s["rouge2"].fmeasure)
        scores["rougeL"].append(s["rougeL"].fmeasure)

    del ds
    gc.collect()

    result = {k: round(sum(v) / len(v), 4) for k, v in scores.items()}
    result["n"] = limit
    print(
        f"    ✓ ROUGE-1={result['rouge1']:.4f}  ROUGE-2={result['rouge2']:.4f}  ROUGE-L={result['rougeL']:.4f}  (n={limit})"
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 4. IberBench — via lm-evaluation-harness (Catalan tasks)
# ──────────────────────────────────────────────────────────────────────────────

IBERBENCH_CATALAN_TASKS = [
    "catcola",  # grammatical acceptability
    "wnli_ca",  # Winograd NLI
    "teca",  # text classification
]


def run_iberbench(
    model_name: str,
    base_url: str | None = None,
    tokenizer: str | None = None,
    n_samples: int | None = None,
) -> dict:
    """
    Runs Catalan IberBench tasks via lm-evaluation-harness.
    Supports llama-server (via base_url) or HF. Requires log-probabilities — not usable with chat APIs.
    """
    if not HAS_LM_EVAL:
        return {"error": "lm_eval not installed"}

    print("\n[6/7] Running IberBench tasks via lm-evaluation-harness …")
    print(f"    Tasks: {', '.join(IBERBENCH_CATALAN_TASKS)}")

    if base_url:
        tok = tokenizer or model_name
        lm_model = "local-completions"
        mistral_fix = (
            ",tokenizer_kwargs={fix_mistral_regex:True}"
            if "mistral" in tok.lower()
            else ""
        )
        lm_model_args = (
            f"model={tok},"
            f"base_url={base_url}/completions,"
            f"tokenizer={tok},"
            f"num_concurrent=1,max_retries=3,tokenized_requests=False,"
            f"add_bos_token=True{mistral_fix}"
        )
    else:
        lm_model = "hf"
        mistral_fix = (
            ",tokenizer_kwargs={fix_mistral_regex:True}"
            if "mistral" in model_name.lower()
            else ""
        )
        lm_model_args = f"pretrained={model_name}{mistral_fix}"

    results = lm_eval.simple_evaluate(
        model=lm_model,
        model_args=lm_model_args,
        tasks=IBERBENCH_CATALAN_TASKS,
        num_fewshot=0,
        batch_size=1,
        log_samples=False,
        limit=n_samples,
        confirm_run_unsafe_code=True,
    )

    scores = {
        task: results["results"][task]
        for task in IBERBENCH_CATALAN_TASKS
        if task in results.get("results", {})
    }
    for task, score in scores.items():
        print(f"    ✓ {task}: {score}")
    return scores


# ──────────────────────────────────────────────────────────────────────────────
# 4. FLORES+ — Machine Translation (English ↔ Catalan)
# ──────────────────────────────────────────────────────────────────────────────


def run_flores(
    model_name: str,
    base_url: str | None = None,
    tokenizer: str | None = None,
    n_samples: int | None = None,
    openai_model: str | None = None,
    gemini_model: str | None = None,
    gemini_api_key: str | None = None,
) -> dict:
    """
    Translation evaluation on FLORES+ devtest split via lm-evaluation-harness.
    Tests: English → Catalan and Catalan → English.
    Metric: BLEU, TER, chrF (computed by lm-eval).
    Supports llama-server (via base_url), OpenAI API (via openai_model), or HF.
    """
    if not HAS_LM_EVAL:
        return {"error": "lm_eval not installed"}

    print("\n[7/7] Running FLORES+ (EN↔CA translation) via lm-evaluation-harness …")

    if gemini_model:
        lm_model = "openai-chat-completions"
        _gemini_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        # max_gen_toks must be large enough for thinking models (gemini-3.x) that
        # consume tokens for internal reasoning before producing the translation.
        # The default of 256 is too small and causes mid-sentence truncation.
        lm_model_args = f"model={gemini_model},base_url={_gemini_base_url},max_gen_toks=2048"
        _orig_api_key = os.environ.get("OPENAI_API_KEY")
        _orig_base_url = os.environ.get("OPENAI_BASE_URL")
        os.environ["OPENAI_API_KEY"] = gemini_api_key or ""
        os.environ["OPENAI_BASE_URL"] = _gemini_base_url
    elif openai_model:
        lm_model = "openai-chat-completions"
        lm_model_args = f"model={openai_model}"
    elif base_url:
        tok = tokenizer or model_name
        lm_model = "local-chat-completions"
        lm_model_args = (
            f"model={tok},"
            f"base_url={base_url}/chat/completions,"
            f"num_concurrent=1,max_retries=3,tokenized_requests=False"
        )
    else:
        lm_model = "hf"
        mistral_fix = (
            ",tokenizer_kwargs={fix_mistral_regex:True}"
            if "mistral" in model_name.lower()
            else ""
        )
        lm_model_args = f"pretrained={model_name}{mistral_fix}"

    flores_tasks = ["catalan_bench_flores_en-ca", "catalan_bench_flores_ca-en"]
    try:
        results = lm_eval.simple_evaluate(
            model=lm_model,
            model_args=lm_model_args,
            tasks=flores_tasks,
            num_fewshot=2,
            apply_chat_template=True,
            fewshot_as_multiturn=True,
            batch_size=1,
            log_samples=False,
            limit=n_samples,
            confirm_run_unsafe_code=True,
        )
    finally:
        if gemini_model:
            if _orig_api_key is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = _orig_api_key
            if _orig_base_url is None:
                os.environ.pop("OPENAI_BASE_URL", None)
            else:
                os.environ["OPENAI_BASE_URL"] = _orig_base_url

    scores = {
        task: results["results"][task]
        for task in flores_tasks
        if task in results.get("results", {})
    }
    for task, score in scores.items():
        bleu = score.get("bleu,none", "n/a")
        print(f"    ✓ {task}: BLEU={bleu}")
    return scores


# ──────────────────────────────────────────────────────────────────────────────
# llama-server lifecycle helpers
# ──────────────────────────────────────────────────────────────────────────────


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


def _is_thinking_model(model_spec: str) -> bool:
    """Return True for models known to emit thinking tokens (e.g. Gemma-4 E4B)."""
    lower = model_spec.lower()
    return "gemma-4" in lower or "gemma4" in lower or "-e4b" in lower


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

    model_base = repo.split("/")[-1].replace("-GGUF", "")
    filename = f"{model_base}-{quant}.gguf"

    print(f"[server] Ensuring {filename} is cached locally …", flush=True)
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    blob_dir = os.path.join(cache_dir, f"models--{repo.replace('/', '--')}", "blobs")
    incomplete = os.path.join(blob_dir, filename + ".incomplete")
    # Check if already fully cached (blob file exists directly)
    blob_path = os.path.join(blob_dir, filename)
    if os.path.exists(blob_path):
        local_path = blob_path
        print(f"[server] Already cached at {local_path}", flush=True)
    else:
        # Download with manual progress
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
            # Re-open for appending; re-request with Range if partial
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
    cmd = [
        "/home/jupyter/llama.cpp/build/bin/llama-server",
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
    # Pass current env (HF_HOME already stripped at startup if /mnt/sda1 missing)
    env = os.environ.copy()
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=log_file,
        env=env,
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


# ──────────────────────────────────────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Catalan LLM evaluation pipeline")
    parser.add_argument(
        "--model",
        default="bartowski/Llama-3.2-3B-Instruct-GGUF:Q8_0",
        help="Model spec: GGUF (e.g. 'bartowski/Llama-3.2-3B-Instruct-GGUF:Q8_0'), 'gemini', or 'openai'",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Google AI API key (required when --model gemini)",
    )
    parser.add_argument(
        "--gemini-model",
        default="gemma-3-27b-it",
        help="Gemini/Gemma model name for API calls (default: gemma-3-27b-it)",
    )
    parser.add_argument(
        "--openai-model",
        default=None,
        help="OpenAI model name for API calls (required when --model openai)",
    )
    parser.add_argument(
        "--openai-base-url",
        default=None,
        help="Base URL for OpenAI-compatible API (e.g. OpenRouter)",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=[
            "veritasqa",
            "sts_ca",
            "catcola",
            "club",
            "casum",
            "iberbench",
            "flores",
            "all",
        ],
        default=["all"],
        help="Which benchmarks to run (default: all)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples per benchmark (default: 100)",
    )
    parser.add_argument(
        "--output",
        default="evals/catalan_eval_results.json",
        help="Output file for results (default: evals/catalan_eval_results.json)",
    )
    parser.add_argument(
        "--llama-server-port",
        type=int,
        default=8080,
        help="Port for llama-server when running IberBench/FLORES with a GGUF model (default: 8080)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for llama-server inference (default: cpu)",
    )
    args = parser.parse_args()

    run_all = "all" in args.benchmarks
    to_run = (
        set(args.benchmarks)
        if not run_all
        else {"veritasqa", "sts_ca", "catcola", "club", "casum", "iberbench", "flores"}
    )

    # ── Validate model spec ───────────────────────────────────────────────────
    if args.model not in ("gemini", "openai") and not _is_gguf_model(args.model):
        raise ValueError(
            f"Only GGUF models are supported. Got: {args.model}\n"
            "Use a GGUF spec like 'bartowski/Llama-3.2-3B-Instruct-GGUF:Q8_0', '--model gemini', or '--model openai'."
        )

    tokenizer_id = (
        _hf_tokenizer_from_gguf(args.model) if _is_gguf_model(args.model) else None
    )

    def _run_benchmarks(model, lm_eval_base_url: str | None = None):
        model_label = args.gemini_model if args.model == "gemini" else (
            args.openai_model if args.model == "openai" else args.model
        )
        results = {"model": model_label, "benchmarks": {}}

        if "veritasqa" in to_run:
            results["benchmarks"]["veritasqa"] = run_veritasqa(model, args.n_samples)

        if "sts_ca" in to_run:
            results["benchmarks"]["sts_ca"] = run_sts_ca(model, args.n_samples)

        if "casum" in to_run:
            results["benchmarks"]["casum"] = run_casum(model, args.n_samples)

        if "catcola" in to_run:
            results["benchmarks"]["catcola"] = run_catcola(model, args.n_samples)

        if "club" in to_run:
            results["benchmarks"]["club_qa"] = run_club_qa(model, args.n_samples)

        if "iberbench" in to_run:
            if args.model in ("gemini", "openai"):
                print(
                    "\n[6/7] IberBench skipped — tasks require log-probabilities not available via chat API."
                )
                results["benchmarks"]["iberbench"] = {
                    "note": "requires llama-server model (log-prob tasks)"
                }
            else:
                try:
                    results["benchmarks"]["iberbench"] = run_iberbench(
                        args.model, lm_eval_base_url, tokenizer_id, args.n_samples,
                    )
                except Exception as e:
                    print(f"[warn] IberBench failed: {e}")
                    results["benchmarks"]["iberbench"] = {"error": str(e)}

        if "flores" in to_run:
            try:
                results["benchmarks"]["flores"] = run_flores(
                    args.model, lm_eval_base_url, tokenizer_id, args.n_samples,
                    openai_model=args.openai_model if args.model == "openai" else None,
                    gemini_model=args.gemini_model if args.model == "gemini" else None,
                    gemini_api_key=args.api_key if args.model == "gemini" else None,
                )
            except Exception as e:
                print(f"[warn] FLORES failed: {e}")
                results["benchmarks"]["flores"] = {"error": str(e)}

        return results

    # ── Run benchmarks ────────────────────────────────────────────────────────
    t_start = time.time()
    if args.model == "gemini":
        if not args.api_key:
            raise ValueError("--api-key is required when using --model gemini")
        model = GeminiModel(api_key=args.api_key, model_name=args.gemini_model)
        results = _run_benchmarks(model)
    elif args.model == "openai":
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required when using --model openai")
        if not args.openai_model:
            raise ValueError("--openai-model is required when using --model openai")
        model = OpenAIModel(
            api_key=openai_api_key,
            model_name=args.openai_model,
            base_url=args.openai_base_url,
        )
        results = _run_benchmarks(model)
    else:
        server_extra = ["--reasoning", "off"] if _is_thinking_model(args.model) else None
        with llama_server_context(
            args.model, args.llama_server_port, args.device, extra_args=server_extra
        ) as base_url:
            model = LlamaServerModel(args.model, base_url)
            results = _run_benchmarks(model, base_url)

    # ── Save & print summary ──────────────────────────────────────────────────
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t_start
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))

    print("\n" + "═" * 60)
    print("  SUMMARY")
    print("═" * 60)
    print(f"  Model : {args.model}")
    for bench, res in results["benchmarks"].items():
        print(f"  {bench:<15} → {res}")
    print(f"  Total time    : {elapsed_str}")
    print("═" * 60)
    print(f"\n  Full results saved to: {output_path}\n")


if __name__ == "__main__":
    main()
