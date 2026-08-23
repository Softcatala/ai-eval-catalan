"""
Catalan Linguistic Competency Evaluation Pipeline
Evaluates GGUF and API models on key Catalan benchmarks:
  1. STS-ca    – semantic textual similarity
  2. CatCoLA   – grammatical acceptability
  3. CLUB      – reading comprehension QA
  4. CaSum     – summarization
  5. FLORES+   – machine translation quality
  6. IFEval-ca – instruction following

Requirements:
  pip install datasets scikit-learn sacrebleu lm_eval huggingface_hub
  llama-server must be available in PATH (from llama.cpp)

Usage:
  # With a llama.cpp GGUF model from bartowski (default):
  python model.py --model "bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M"

  # With the Google AI API (Gemma 4):
  python model.py --model "gemini" --api-key "YOUR_KEY"

  # With Claude via OpenRouter:
  python model.py --model "claude" --api-key "YOUR_OPENROUTER_KEY" --openai-model "anthropic/claude-opus-4-7"

  # Run only specific benchmarks:
  python model.py --model "bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M" --benchmarks catcola flores
"""

import argparse
from collections import Counter
import gc
import json
import math
import os
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from comet_config import COMET_CHECKPOINT, drop_legacy_translation_metrics
from inference import chat_completion_params, lm_eval_params

from llamaserver import (
    LlamaServerModel,
    _hf_tokenizer_from_gguf,
    _is_gguf_model,
)

# facebook/flores uses an old dataset script — trust it so datasets doesn't refuse to load it.
os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")

DEFAULT_LOCAL_SERVER_URL = "http://localhost:9090/v1"

from datasets import load_dataset
from sklearn.metrics import matthews_corrcoef

# ── Optional: lm_eval for lm-evaluation-harness tasks ─────────────────────────
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

    # Gemini's OpenAI-compatible endpoint rejects the optional `seed` field that
    # lm-eval unconditionally adds to chat-completion payloads.  Keep lm-eval's
    # normal payload everywhere else and strip only that unsupported field for
    # Google's endpoint.
    _original_openai_chat_payload = _lm_oai.OpenAIChatCompletion._create_payload

    def _gemini_compatible_chat_payload(self, *args, **kwargs):
        payload = _original_openai_chat_payload(self, *args, **kwargs)
        if "generativelanguage.googleapis.com" in self.base_url:
            payload.pop("seed", None)
            # lm-eval emits max_gen_toks, but Gemini's OpenAI endpoint expects max_completion_tokens.
            max_gen_toks = payload.pop("max_gen_toks", None)
            if max_gen_toks is not None and "max_completion_tokens" not in payload:
                payload["max_completion_tokens"] = max_gen_toks
        return payload

    _lm_oai.OpenAIChatCompletion._create_payload = _gemini_compatible_chat_payload
    HAS_LM_EVAL = True
except ImportError:
    HAS_LM_EVAL = False
    print("[warn] lm_eval not found – lm-evaluation-harness tasks will be skipped.")
    print("       Install with: pip install lm_eval")


# ──────────────────────────────────────────────────────────────────────────────
# Model wrapper — supports llama-server (GGUF) and Google AI API
# ──────────────────────────────────────────────────────────────────────────────



class GeminiModel:
    """Google AI API wrapper (for Gemma 4 / Gemini models)."""

    def __init__(self, api_key: str, model_name: str = "gemma-3-27b-it"):
        from openai import OpenAI

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        self.model_name = model_name

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        params = chat_completion_params(max_new_tokens, "gemini", self.model_name)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **params,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"[error] API call failed: {e}")
            time.sleep(2)
            return ""



class OpenAIModel:
    """OpenAI-compatible API wrapper (works with OpenAI and OpenRouter)."""

    def __init__(self, api_key: str, model_name: str, base_url: str | None = None):
        from openai import OpenAI

        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.provider = "openrouter" if base_url and "openrouter" in base_url else "openai"
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_cost: float | None = None  # None until first response with cost data

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        try:
            params = chat_completion_params(max_new_tokens, self.provider, self.model_name)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **params,
            )
            if response.usage:
                self.prompt_tokens += response.usage.prompt_tokens
                self.completion_tokens += response.usage.completion_tokens
                # OpenRouter returns cost directly in usage
                call_cost = getattr(response.usage, "cost", None)
                if call_cost is not None:
                    self.total_cost = (self.total_cost or 0.0) + call_cost
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"[error] API call failed: {e}")
            time.sleep(2)
            return ""


# ──────────────────────────────────────────────────────────────────────────────
# 1. STS-ca — Semantic Textual Similarity (Paraphrase proxy)
# ──────────────────────────────────────────────────────────────────────────────


def run_sts_ca(model, n_samples: int = 100) -> dict:
    """
    Semantic textual similarity as a proxy for paraphrase quality.
    The model is given two sentences and asked to rate their similarity (0–5).
    Metric: Pearson correlation between predicted and gold scores.
    Dataset: projecte-aina/sts-ca (test split).
    """
    print("\n[1/6] Running STS-ca (paraphrase / similarity) …")
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
# 2. CatCoLA — Catalan Corpus of Linguistic Acceptability
# ──────────────────────────────────────────────────────────────────────────────


def parse_catcola_answer(answer: str) -> int | None:
    """Parse CatCoLA yes/no answers. Return None for format failures."""
    text = answer.strip().lower()
    if re.match(r"^(sí|si)\b", text):
        return 1
    if re.match(r"^no\b", text):
        return 0
    return None


def run_catcola(model, n_samples: int = 200) -> dict:
    """
    Binary classification: is a Catalan sentence grammatically acceptable?
    Metric: Matthews Correlation Coefficient (MCC), as per CoLA standard.
    Dataset: nbel/CatCoLA on HuggingFace.
    """
    print("\n[2/6] Running CatCoLA …")
    ds = load_dataset("nbel/CatCoLA", split="validation")
    limit = min(n_samples, len(ds))

    preds, labels = [], []
    strict_correct = []
    invalid = 0
    for i in range(limit):
        item = ds[i]
        sentence = item["Sentence"]
        label = item["Label"]  # 0 = unacceptable, 1 = acceptable

        prompt = (
            "La seguent frase en catala es gramaticalment correcta? "
            "Respon nomes amb 'si' o 'no'.\n\n"
            f"Frase: {sentence}\nResposta:"
        )
        answer = model.generate(prompt, max_new_tokens=16)
        pred = parse_catcola_answer(answer)

        if pred is None:
            invalid += 1
            strict_correct.append(False)
            continue

        preds.append(pred)
        labels.append(label)
        strict_correct.append(pred == label)

    del ds
    gc.collect()

    mcc = matthews_corrcoef(labels, preds) if preds else None
    acc = sum(strict_correct) / limit if limit else 0.0
    invalid_rate = invalid / limit if limit else 0.0
    coverage = len(preds) / limit if limit else 0.0
    result = {
        "mcc": round(mcc, 4) if mcc is not None else None,
        "accuracy": round(acc, 4),
        "invalid_rate": round(invalid_rate, 4),
        "coverage": round(coverage, 4),
        "n": limit,
        "n_valid": len(preds),
        "n_invalid": invalid,
    }
    mcc_text = f"{mcc:.4f}" if mcc is not None else "n/a"
    print(
        f"    ✓ MCC(valid)={mcc_text}  Strict Accuracy={acc:.4f}  "
        f"Invalid={invalid_rate:.2%}  (n={limit}, valid={len(preds)})"
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 3. CLUB — Catalan Language Understanding Benchmark (QA slice)
# ──────────────────────────────────────────────────────────────────────────────


def run_club_qa(model, n_samples: int = 100) -> dict:
    """
    Extractive QA on VilaQuAD (Catalan Wikipedia QA).
    Metric: Exact Match (EM) and token-level F1.
    Dataset: projecte-aina/vilaquad on HuggingFace.
    """
    print("\n[3/6] Running CLUB / VilaQuAD (QA) …")
    ds = load_dataset("projecte-aina/vilaquad", split="validation")

    # XQuAD Spanish/German/Arabic use a shared SQuAD-style QA-F1 scorer.
    # Mirror that here with case/accent/punctuation/space normalization only.
    def _normalize_answer(text: str) -> str:
        text = re.sub(r"<\|[^|]+_TOKEN\|>", " ", text)
        text = text.lower()
        text = "".join(
            ch for ch in unicodedata.normalize("NFD", text)
            if unicodedata.category(ch) != "Mn"
        )
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
        return " ".join(text.split())

    def _token_f1(prediction: str, gold_answer: str) -> float:
        pred_tokens = _normalize_answer(prediction).split()
        gold_tokens = _normalize_answer(gold_answer).split()
        if not pred_tokens or not gold_tokens:
            return float(pred_tokens == gold_tokens)

        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        return 2 * precision * recall / (precision + recall)

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

    exact_match_total = 0
    token_f1_total = 0.0
    n = 0
    for context, question, gold_answers in _iter_qa_pairs(ds, n_samples):
        prompt = (
            f"Llegeix el text i respon només amb el fragment mínim del text que "
            f"respon la pregunta. No expliquis res més.\n\n"
            f"Text: {context[:800]}\n\nPregunta: {question}\nResposta:"
        )
        raw_pred = model.generate(prompt, max_new_tokens=64).strip()
        if not raw_pred:
            raise RuntimeError("CLUB QA generation returned an empty response")
        pred = raw_pred.lower()
        em = any(gold.strip().lower() in pred for gold in gold_answers)
        f1 = max(_token_f1(raw_pred, gold) for gold in gold_answers)
        exact_match_total += int(em)
        token_f1_total += f1
        n += 1

    del ds
    gc.collect()

    score = exact_match_total / n
    f1_score = token_f1_total / n
    result = {
        "exact_match_approx": round(score, 4),
        "token_f1": round(f1_score, 4),
        "n": n,
    }
    print(
        f"    ✓ Approx. Exact Match={score:.4f}  "
        f"Token F1={f1_score:.4f}  "
        f"(n={n})"
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 4. CaSum — Catalan Summarization
# ──────────────────────────────────────────────────────────────────────────────


def run_casum(model, n_samples: int = 100) -> dict:
    """
    Abstractive summarization on CaSum (Catalan news articles → headline).
    Metric: ROUGE-1, ROUGE-2, ROUGE-L F1.
    Dataset: projecte-aina/casum on HuggingFace (test split).
    """
    from rouge_score import rouge_scorer

    print("\n[4/6] Running CaSum (summarization) …")
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
# 5. FLORES+ — Machine Translation (English/Spanish ↔ Catalan)
# ──────────────────────────────────────────────────────────────────────────────


_FLORES_SOURCE_FIELDS = {
    "en": ("sentence_eng_Latn", "eng_Latn", "en", "english"),
    "es": ("sentence_spa_Latn", "spa_Latn", "es", "spanish"),
    "ca": ("sentence_cat_Latn", "cat_Latn", "ca", "catalan"),
}


def _load_comet_model():
    """Load the standard reference-based WMT22 COMET checkpoint lazily."""
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError as exc:
        raise RuntimeError(
            "COMET is required for FLORES scoring; install project dependencies"
        ) from exc
    return load_from_checkpoint(download_model(COMET_CHECKPOINT))


def _sample_text(value) -> str | None:
    """Unwrap the list-shaped response/target values emitted by lm-eval."""
    while isinstance(value, (list, tuple)) and value:
        value = value[0]
    return value.strip() if isinstance(value, str) and value.strip() else None


def _comet_example(task: str, sample: dict) -> dict[str, str] | None:
    """Convert one logged lm-eval FLORES sample to COMET's input schema."""
    direction = task.removeprefix("catalan_bench_flores_")
    source_language = direction.split("-", 1)[0]
    doc = sample.get("doc", {})
    source = next(
        (
            _sample_text(doc.get(field))
            for field in _FLORES_SOURCE_FIELDS.get(source_language, ())
            if _sample_text(doc.get(field)) is not None
        ),
        None,
    )
    source = source or _sample_text(doc.get("source")) or _sample_text(doc.get("src"))
    reference = _sample_text(sample.get("target"))
    hypothesis = _sample_text(sample.get("filtered_resps")) or _sample_text(
        sample.get("resps")
    )
    if not all((source, reference, hypothesis)):
        return None
    return {"src": source, "ref": reference, "mt": hypothesis}


def _score_flores_comet(results: dict, tasks: list[str], comet_model) -> dict:
    """Attach COMET scores without discarding successful FLORES directions."""
    scores = {
        task: results["results"][task]
        for task in tasks
        if task in results.get("results", {})
    }
    for task, score in list(scores.items()):
        examples = [
            example
            for sample in results.get("samples", {}).get(task, [])
            if (example := _comet_example(task, sample)) is not None
        ]
        try:
            if not examples:
                raise RuntimeError("lm-eval returned no usable translations for COMET")
            prediction = comet_model.predict(
                examples,
                batch_size=8,
                # llama.cpp owns the GPU in this pipeline. Run COMET on CPU so its
                # PyTorch/CUDA build cannot conflict with the host driver.
                gpus=0,
            )
        except Exception as exc:
            print(f"    [warn] {task}: COMET failed: {exc}")
            scores[task] = {"error": str(exc)}
            continue

        score["n"] = len(examples)
        score["comet"] = float(prediction.system_score)
        # BLEU is still produced internally by the lm-eval FLORES task, but it
        # is no longer part of our result contract.
        drop_legacy_translation_metrics(score)
        print(f"    ✓ {task}: COMET={score['comet']:.4f}")
    return scores


def run_flores(
    model_name: str,
    base_url: str | None = None,
    tokenizer: str | None = None,
    n_samples: int | None = None,
    openai_model: str | None = None,
    gemini_model: str | None = None,
    gemini_api_key: str | None = None,
    openrouter_model: str | None = None,
    openrouter_api_key: str | None = None,
    tasks: list[str] | None = None,
) -> dict:
    """
    Translation evaluation on FLORES+ devtest split via lm-evaluation-harness.
    Tests both directions for English↔Catalan and Spanish↔Catalan.
    Metric: COMET (computed from lm-eval's generated translations).
    Supports llama-server (via base_url), OpenAI API (via openai_model), OpenRouter, or HF.
    """
    if not HAS_LM_EVAL:
        return {"error": "lm_eval not installed"}

    print("\n[5/6] Running FLORES+ (EN↔CA and ES↔CA translation) via lm-evaluation-harness …")

    _openrouter_base_url = "https://openrouter.ai/api/v1"
    bedrock_anthropic = bool(
        openai_model
        and base_url
        and "bedrock-mantle" in base_url
        and openai_model.startswith("anthropic.")
    )
    inference_provider = "anthropic" if bedrock_anthropic else "gemini" if gemini_model else "openrouter" if openrouter_model else "openai" if openai_model else "llama" if base_url else "hf"
    inference_model = gemini_model or openrouter_model or openai_model or model_name

    if gemini_model:
        lm_model = "openai-chat-completions"
        _gemini_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        lm_model_args = (
            f"model={gemini_model},base_url={_gemini_base_url},"
            f"eos_string=</s>,num_concurrent=8,max_retries=3,timeout=120"
        )
        _orig_api_key = os.environ.get("OPENAI_API_KEY")
        _orig_base_url = os.environ.get("OPENAI_BASE_URL")
        os.environ["OPENAI_API_KEY"] = gemini_api_key or ""
        os.environ["OPENAI_BASE_URL"] = _gemini_base_url
    elif openrouter_model:
        lm_model = "openai-chat-completions"
        lm_model_args = (
            f"model={openrouter_model},"
            f"base_url={_openrouter_base_url}/chat/completions,"
            f"api_key={openrouter_api_key},"
            f"num_concurrent=1,max_retries=3,tokenized_requests=False"
        )
        _orig_api_key = os.environ.get("OPENAI_API_KEY")
        _orig_base_url = os.environ.get("OPENAI_BASE_URL")
        os.environ["OPENAI_API_KEY"] = openrouter_api_key or ""
        os.environ["OPENAI_BASE_URL"] = _openrouter_base_url
    elif bedrock_anthropic:
        # Claude Opus 4.7 does not implement Bedrock's OpenAI-compatible Chat
        # Completions API.  It supports the native Anthropic Messages endpoint
        # on Bedrock Mantle instead.  lm-eval has an Anthropic adapter, and this
        # small variant supplies Bedrock's bearer authentication and removes
        # ``temperature``, which Opus 4.7 has deprecated.
        from lm_eval.models.anthropic_llms import AnthropicChat

        class BedrockAnthropicChat(AnthropicChat):
            @property
            def header(self):
                token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
                if not token:
                    raise ValueError("AWS_BEARER_TOKEN_BEDROCK is required")
                return {
                    "Authorization": f"Bearer {token}",
                    "anthropic-version": "bedrock-2023-05-31",
                }

            def _create_payload(self, *args, **kwargs):
                payload = super()._create_payload(*args, **kwargs)
                payload.pop("temperature", None)
                payload.pop("reasoning_effort", None)
                return payload

            @staticmethod
            def parse_generations(outputs, **kwargs):
                if not isinstance(outputs, list):
                    outputs = [outputs]
                return [
                    "".join(
                        block.get("text", "")
                        for block in output.get("content", [])
                        if isinstance(block, dict)
                    )
                    for output in outputs
                ]

        lm_model = BedrockAnthropicChat(
            model=openai_model,
            base_url=f"{base_url.rstrip('/').removesuffix('/v1')}/anthropic/v1/messages",
            tokenizer_backend="none",
            tokenized_requests=False,
            num_concurrent=1,
            max_retries=3,
        )
        lm_model_args = None
    elif openai_model:
        lm_model = "openai-chat-completions"
        _base = f"base_url={base_url}/chat/completions," if base_url else ""
        lm_model_args = (
            f"model={openai_model},"
            f"{_base}"
            f"num_concurrent=1,max_retries=3,tokenized_requests=False"
        )
    elif base_url:
        lm_model = "local-chat-completions"
        lm_model_args = (
            f"model={model_name},"
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

    gen_kwargs = lm_eval_params(2048, inference_provider, inference_model)

    flores_tasks = tasks or [
        "catalan_bench_flores_en-ca",
        "catalan_bench_flores_ca-en",
        "catalan_bench_flores_es-ca",
        "catalan_bench_flores_ca-es",
    ]
    try:
        results = lm_eval.simple_evaluate(
            model=lm_model,
            model_args=lm_model_args,
            tasks=flores_tasks,
            num_fewshot=2,
            apply_chat_template=True,
            fewshot_as_multiturn=True,
            batch_size=1,
            # COMET needs the source, reference, and generated translation for
            # every example. lm-eval returns those only when samples are logged.
            log_samples=True,
            limit=n_samples,
            bootstrap_iters=0,
            confirm_run_unsafe_code=True,
            gen_kwargs=gen_kwargs,
        )
    finally:
        if gemini_model or openrouter_model:
            if _orig_api_key is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = _orig_api_key
            if _orig_base_url is None:
                os.environ.pop("OPENAI_BASE_URL", None)
            else:
                os.environ["OPENAI_BASE_URL"] = _orig_base_url
    comet_model = _load_comet_model()
    return _score_flores_comet(results, flores_tasks, comet_model)



# ──────────────────────────────────────────────────────────────────────────────
# 6. IFEval-ca — Instruction Following (Catalan)
# ──────────────────────────────────────────────────────────────────────────────


def run_ifeval(
    model_name: str,
    base_url: str | None = None,
    tokenizer: str | None = None,
    n_samples: int | None = None,
    openai_model: str | None = None,
    gemini_model: str | None = None,
    gemini_api_key: str | None = None,
    openrouter_model: str | None = None,
    openrouter_api_key: str | None = None,
) -> dict:
    """
    Catalan IFEval — instruction-following evaluation via lm-evaluation-harness.
    Dataset: projecte-aina/IFEval_ca (541 prompts professionally translated from Google IFEval).
    Metric: prompt-level / instruction-level strict & loose accuracy (rule-based, deterministic).
    Generative task — works with chat APIs and local llama-server (no log-probs required).
    """
    if not HAS_LM_EVAL:
        return {"error": "lm_eval not installed"}

    print("\n[6/6] Running IFEval-ca (instruction following) via lm-evaluation-harness …")

    _openrouter_base_url = "https://openrouter.ai/api/v1"
    _orig_api_key = None
    _orig_base_url = None
    needs_env_restore = False
    inference_provider = "gemini" if gemini_model else "openrouter" if openrouter_model else "openai" if openai_model else "llama" if base_url else "hf"
    inference_model = gemini_model or openrouter_model or openai_model or model_name

    if gemini_model:
        lm_model = "openai-chat-completions"
        _gemini_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        # eos_string=</s> avoids lm_eval sending an empty stop=[] array which the
        # Gemini OpenAI-compat endpoint rejects with 400 "Value is not a string: []".
        # The literal string is unlikely to appear in Catalan output, so generation
        # will end naturally on max_gen_toks instead.
        lm_model_args = (
            f"model={gemini_model},base_url={_gemini_base_url},"
            f"eos_string=</s>,num_concurrent=8,max_retries=3,timeout=120"
        )
        _orig_api_key = os.environ.get("OPENAI_API_KEY")
        _orig_base_url = os.environ.get("OPENAI_BASE_URL")
        os.environ["OPENAI_API_KEY"] = gemini_api_key or ""
        os.environ["OPENAI_BASE_URL"] = _gemini_base_url
        needs_env_restore = True
    elif openrouter_model:
        lm_model = "openai-chat-completions"
        lm_model_args = (
            f"model={openrouter_model},"
            f"base_url={_openrouter_base_url}/chat/completions,"
            f"api_key={openrouter_api_key},"
            f"num_concurrent=4,max_retries=3,timeout=120,tokenized_requests=False"
        )
        _orig_api_key = os.environ.get("OPENAI_API_KEY")
        _orig_base_url = os.environ.get("OPENAI_BASE_URL")
        os.environ["OPENAI_API_KEY"] = openrouter_api_key or ""
        os.environ["OPENAI_BASE_URL"] = _openrouter_base_url
        needs_env_restore = True
    elif openai_model:
        lm_model = "openai-chat-completions"
        _base = f"base_url={base_url}/chat/completions," if base_url else ""
        lm_model_args = (
            f"model={openai_model},"
            f"{_base}"
            f"num_concurrent=8,max_retries=3,timeout=120,tokenized_requests=False"
        )
    elif base_url:
        lm_model = "local-chat-completions"
        lm_model_args = (
            f"model={model_name},"
            f"base_url={base_url}/chat/completions,"
            f"num_concurrent=1,max_retries=3,timeout=120,tokenized_requests=False"
        )
    else:
        lm_model = "hf"
        mistral_fix = (
            ",tokenizer_kwargs={fix_mistral_regex:True}"
            if "mistral" in model_name.lower()
            else ""
        )
        lm_model_args = f"pretrained={model_name}{mistral_fix}"

    gen_kwargs = lm_eval_params(2048, inference_provider, inference_model)

    try:
        results = lm_eval.simple_evaluate(
            model=lm_model,
            model_args=lm_model_args,
            tasks=["ifeval_ca"],
            num_fewshot=0,
            apply_chat_template=True,
            batch_size=1,
            log_samples=False,
            limit=n_samples,
            confirm_run_unsafe_code=True,
            gen_kwargs=gen_kwargs,
        )
    finally:
        if needs_env_restore:
            if _orig_api_key is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = _orig_api_key
            if _orig_base_url is None:
                os.environ.pop("OPENAI_BASE_URL", None)
            else:
                os.environ["OPENAI_BASE_URL"] = _orig_base_url

    score = results["results"].get("ifeval_ca", {})
    score["n"] = n_samples
    p_strict = score.get("prompt_level_strict_acc,none", "n/a")
    i_strict = score.get("inst_level_strict_acc,none", "n/a")
    p_loose = score.get("prompt_level_loose_acc,none", "n/a")
    i_loose = score.get("inst_level_loose_acc,none", "n/a")
    print(
        f"    ✓ prompt-strict={p_strict}  inst-strict={i_strict}  "
        f"prompt-loose={p_loose}  inst-loose={i_loose}"
    )
    return score


# ──────────────────────────────────────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Catalan LLM evaluation pipeline")
    parser.add_argument(
        "--model",
        default="bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M",
        help="Model spec: GGUF (e.g. 'bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M'), 'gemini', 'openai', or 'claude'",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key: Google AI key for --model gemini, OpenRouter key for --model claude",
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
            "sts_ca",
            "catcola",
            "club",
            "casum",
            "flores",
            "ifeval",
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
        "--flores-tasks",
        nargs="+",
        choices=[
            "catalan_bench_flores_en-ca",
            "catalan_bench_flores_ca-en",
            "catalan_bench_flores_es-ca",
            "catalan_bench_flores_ca-es",
        ],
        default=None,
        help="Optional FLORES direction subset (default: all four directions)",
    )
    parser.add_argument(
        "--output",
        default="evals/catalan_eval_results.json",
        help="Output file for results (default: evals/catalan_eval_results.json)",
    )
    parser.add_argument(
        "--llama-server-url",
        "--server-url",
        dest="llama_server_url",
        default=os.environ.get("LLAMA_SERVER_URL", DEFAULT_LOCAL_SERVER_URL),
        help="Existing llama-server OpenAI-compatible base URL (e.g. http://localhost:9090/v1). Also configurable with LLAMA_SERVER_URL.",
    )
    parser.add_argument(
        "--llama-server-model",
        "--server-model",
        dest="llama_server_model",
        default=None,
        help="Model id to send to an existing multi-model llama-server (defaults to --model)",
    )
    parser.add_argument(
        "--params-b",
        type=float,
        default=None,
        help="Model parameter count in billions (e.g. 7.0 for a 7B model)",
    )
    parser.add_argument(
        "--display-name",
        default=None,
        help="Name to use in tables and charts (defaults to model identifier)",
    )
    parser.add_argument(
        "--cloud",
        action="store_true",
        default=False,
        help="Mark this model as a cloud API model (not a local GGUF)",
    )
    parser.add_argument(
        "--quantized-analysis",
        action="store_true",
        default=False,
        help="Mark this model as a quantized (Q4) variant for analysis purposes",
    )
    args = parser.parse_args()

    # ── Compute memory estimate ───────────────────────────────────────────────
    def _estimate_memory_gb(params_b: float | None, model_spec: str) -> float | None:
        """Estimate VRAM usage in GB from parameter count and quantization type."""
        if params_b is None:
            return None
        # bits per parameter for common GGUF quantization levels
        _BITS = {
            "Q8_0": 8.5,
            "UD-Q8_K_XL": 8.5,
            "Q4_K_M": 4.5,
            "UD-Q4_K_XL": 4.5,
            "Q4_0": 4.5,
            "Q5_K_M": 5.5,
            "Q6_K": 6.5,
        }
        quant = model_spec.rsplit(":", 1)[-1] if ":" in model_spec else "Q4_K_M"
        bits = _BITS.get(quant, 8.5)
        return round(params_b * 1e9 * bits / 8 / 1e9, 1)

    run_all = "all" in args.benchmarks
    to_run = (
        set(args.benchmarks)
        if not run_all
        else {"sts_ca", "catcola", "club", "casum", "flores", "ifeval"}
    )

    # ── Validate model spec ───────────────────────────────────────────────────
    if args.model not in ("gemini", "openai", "claude") and not _is_gguf_model(args.model):
        raise ValueError(
            f"Only GGUF models are supported. Got: {args.model}\n"
            "Use a GGUF spec like 'bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M', '--model gemini', '--model openai', or '--model claude'."
        )

    tokenizer_id = (
        _hf_tokenizer_from_gguf(args.model) if _is_gguf_model(args.model) else None
    )

    def _run_benchmarks(model, lm_eval_base_url: str | None = None):
        model_label = args.gemini_model if args.model == "gemini" else (
            args.openai_model if args.model in ("openai", "claude") else args.model
        )
        lm_eval_model_name = args.llama_server_model or args.model
        memory_gb = _estimate_memory_gb(args.params_b, args.model)
        results = {
            "model": model_label,
            "display_name": args.display_name,
            "cloud": args.cloud,
            "params_b": args.params_b,
            "memory_gb": memory_gb,
            "evaluated_at": datetime.fromtimestamp(
                t_start, timezone.utc
            ).isoformat(timespec="seconds"),
            "benchmarks": {},
        }

        if "sts_ca" in to_run:
            results["benchmarks"]["sts_ca"] = run_sts_ca(model, args.n_samples)

        if "casum" in to_run:
            results["benchmarks"]["casum"] = run_casum(model, args.n_samples)

        if "catcola" in to_run:
            results["benchmarks"]["catcola"] = run_catcola(model, args.n_samples)

        if "club" in to_run:
            results["benchmarks"]["club_qa"] = run_club_qa(model, args.n_samples)

        if "flores" in to_run:
            try:
                results["benchmarks"]["flores"] = run_flores(
                    lm_eval_model_name, lm_eval_base_url, tokenizer_id, args.n_samples,
                    openai_model=args.openai_model if args.model == "openai" else None,
                    gemini_model=args.gemini_model if args.model == "gemini" else None,
                    gemini_api_key=args.api_key if args.model == "gemini" else None,
                    openrouter_model=args.openai_model if args.model == "claude" else None,
                    openrouter_api_key=(args.api_key or os.environ.get("OPENROUTER_API_KEY")) if args.model == "claude" else None,
                    tasks=args.flores_tasks,
                )
                results["flores_comet_checkpoint"] = COMET_CHECKPOINT
            except Exception as e:
                print(f"[warn] FLORES failed: {e}")
                results["benchmarks"]["flores"] = {"error": str(e)}

        if "ifeval" in to_run:
            try:
                results["benchmarks"]["ifeval"] = run_ifeval(
                    lm_eval_model_name, lm_eval_base_url, tokenizer_id, args.n_samples,
                    openai_model=args.openai_model if args.model == "openai" else None,
                    gemini_model=args.gemini_model if args.model == "gemini" else None,
                    gemini_api_key=args.api_key if args.model == "gemini" else None,
                    openrouter_model=args.openai_model if args.model == "claude" else None,
                    openrouter_api_key=(args.api_key or os.environ.get("OPENROUTER_API_KEY")) if args.model == "claude" else None,
                )
            except Exception as e:
                print(f"[warn] IFEval-ca failed: {e}")
                results["benchmarks"]["ifeval"] = {"error": str(e)}

        return results

    # ── Run benchmarks ────────────────────────────────────────────────────────
    t_start = time.time()
    if args.model == "gemini":
        if not args.api_key:
            raise ValueError("--api-key is required when using --model gemini")
        model = GeminiModel(api_key=args.api_key, model_name=args.gemini_model)
        results = _run_benchmarks(model)
    elif args.model == "claude":
        openrouter_api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise ValueError("--api-key or OPENROUTER_API_KEY is required when using --model claude")
        model = OpenAIModel(
            api_key=openrouter_api_key,
            model_name=args.openai_model or "anthropic/claude-sonnet-4-5",
            base_url="https://openrouter.ai/api/v1",
        )
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
        results = _run_benchmarks(model,args.openai_base_url)
    else:
        if not args.llama_server_url:
            raise ValueError(
                "--server-url or LLAMA_SERVER_URL is required for local GGUF models"
            )
        model = LlamaServerModel(
            args.model,
            args.llama_server_url,
            request_model=args.llama_server_model or args.model,
        )
        results = _run_benchmarks(model, args.llama_server_url.rstrip("/"))

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
    if isinstance(model, OpenAIModel) and (model.prompt_tokens or model.completion_tokens):
        total_tokens = model.prompt_tokens + model.completion_tokens
        print(f"  Tokens        : {model.prompt_tokens:,} in + {model.completion_tokens:,} out = {total_tokens:,} total")
        if model.total_cost is not None:
            print(f"  Cost          : ${model.total_cost:.4f}")
    print("═" * 60)
    print(f"\n  Full results saved to: {output_path}\n")


if __name__ == "__main__":
    main()
