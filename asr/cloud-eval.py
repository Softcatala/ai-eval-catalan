#!/usr/bin/env python3
"""
WER Evaluation Script for cloud ASR APIs.
Evaluates Word Error Rate on FLEURS dataset for Catalan.
Writes results to a JSON file with the same structure as hf-eval.py.

Usage:
    python cloud-eval.py gpt-4o-transcribe --output evals/gpt4o_transcribe.json
    python cloud-eval.py --list-models
"""

import argparse
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from jiwer import wer, cer
from asr_eval_common import hardware_string, load_manifest, normalize_text
from result_io import write_json
from tqdm import tqdm


@dataclass
class EvalResult:
    language: str
    num_samples: int
    wer: float
    cer: float
    total_time: float
    avg_rtf: float


DEFAULT_MANIFEST = Path(__file__).parent / "benchmarks/fleurs_ca_test_400/manifest.json"


LANGUAGE_CONFIG = {
    "ca": {
        "name": "Catalan",
        "fleurs_locale": "ca_es",
        "lang": "catalan",
    },
}

OPENAI_ASR_MODELS = [
    "gpt-4o-transcribe",
]

GEMINI_ASR_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.1-pro-preview",
]

ALL_MODELS = OPENAI_ASR_MODELS + GEMINI_ASR_MODELS


class OpenAIASRWrapper:
    def __init__(self, model_name: str):
        import openai

        self.client = openai.OpenAI()
        self.model_name = model_name

    def transcribe(self, waveform: torch.Tensor, sample_rate: int, lang: str) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            sf.write(tmp_path, waveform.numpy(), sample_rate)
            with open(tmp_path, "rb") as f:
                response = self.client.audio.transcriptions.create(
                    model=self.model_name,
                    file=f,
                    language="ca",
                )
            return response.text
        finally:
            os.unlink(tmp_path)


class GeminiASRWrapper:
    PROMPT = (
        "Transcribe the following Catalan speech segment verbatim into Catalan. "
        "Do not translate. Do not paraphrase. Do not add any commentary.\n"
        "Formatting rules:\n"
        "* Output only the transcription, nothing else, with no newlines.\n"
        "* Do not add punctuation unless it was clearly spoken.\n"
        "* Write numbers as digits (e.g. 3 not three, 1.7 not one point seven)."
    )

    def __init__(self, model_name: str):
        from google import genai
        from google.genai import types

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        self.client = genai.Client(api_key=api_key)
        self.types = types
        self.model_name = model_name

    def transcribe(self, waveform: torch.Tensor, sample_rate: int, lang: str) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            sf.write(tmp_path, waveform.numpy(), sample_rate)
            uploaded = self.client.files.upload(
                file=tmp_path, config={"mime_type": "audio/wav"}
            )
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    self.types.Part.from_uri(
                        file_uri=uploaded.uri, mime_type="audio/wav"
                    ),
                    self.PROMPT,
                ],
                config=self.types.GenerateContentConfig(
                    temperature=1.0, max_output_tokens=2048
                ),
            )
            self.client.files.delete(name=uploaded.name)
            return response.text.strip() if response.text else ""
        finally:
            os.unlink(tmp_path)


def load_model(model_name: str):
    if model_name in OPENAI_ASR_MODELS:
        print(f"Loading OpenAI ASR model: {model_name}")
        return OpenAIASRWrapper(model_name)
    elif model_name in GEMINI_ASR_MODELS:
        print(f"Loading Gemini ASR model: {model_name}")
        return GeminiASRWrapper(model_name)
    else:
        raise ValueError(f"Unknown model: {model_name}. Available: {ALL_MODELS}")


def evaluate_language(
    model,
    model_name: str,
    manifest_path: Path,
    warmup: int = 3,
) -> EvalResult:
    manifest = load_manifest(manifest_path)
    lang_config = LANGUAGE_CONFIG["ca"]
    model_lang = lang_config["lang"]

    print(f"\n{'=' * 60}")
    print(f"Evaluating {lang_config['name']} (ca) on FLEURS")
    print(f"Model: {model_name} | Lang code: {model_lang}")
    print(f"Manifest: {manifest_path} ({manifest['sha256'][:12]})")
    print(f"{'=' * 60}")

    print(f"Evaluating {len(manifest['records'])} local samples...")

    references = []
    hypotheses = []
    rtfs = []
    skipped = 0
    processed = 0
    start_time = time.time()

    with torch.no_grad():
        for record in tqdm(
            manifest["records"], desc=f"Processing {lang_config['name']}"
        ):
            try:
                audio_path = manifest_path.parent / record["audio"]
                audio_array, sample_rate = sf.read(audio_path, dtype="float32")
                if audio_array.ndim == 2:
                    audio_array = audio_array.mean(axis=1)
                duration = len(audio_array) / sample_rate

                waveform = torch.tensor(audio_array, dtype=torch.float32)

                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                    waveform = resampler(waveform.unsqueeze(0)).squeeze(0)
                    sample_rate = 16000

                inference_start = time.perf_counter()
                hypothesis = model.transcribe(waveform, sample_rate, model_lang)
                inference_end = time.perf_counter()

                if processed >= warmup:
                    rtf = (inference_end - inference_start) / duration
                    rtfs.append(rtf)

                ref_normalized = normalize_text(record["reference"])
                hyp_normalized = normalize_text(hypothesis)

                if ref_normalized:
                    references.append(ref_normalized)
                    hypotheses.append(hyp_normalized)

                processed += 1

            except Exception as e:
                print(f"\nError processing sample: {e}")
                skipped += 1
                continue

    total_time = time.time() - start_time

    if references:
        word_error_rate = wer(references, hypotheses)
        char_error_rate = cer(references, hypotheses)
    else:
        word_error_rate = 1.0
        char_error_rate = 1.0

    avg_rtf = float(np.mean(rtfs)) if rtfs else 0.0

    result = EvalResult(
        language=lang_config["name"],
        num_samples=len(references),
        wer=word_error_rate,
        cer=char_error_rate,
        total_time=total_time,
        avg_rtf=avg_rtf,
    )

    print(f"\nResults for {lang_config['name']}:")
    print(f"  Samples: {result.num_samples} (skipped: {skipped})")
    print(f"  WER: {result.wer:.2%} | CER: {result.cer:.2%}")
    print(
        f"  RTF: {result.avg_rtf:.3f} ({1 / result.avg_rtf:.1f}x real-time)"
        if result.avg_rtf > 0
        else "  RTF: N/A"
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a cloud ASR model on FLEURS and write a JSON results file"
    )
    parser.add_argument(
        "model",
        type=str,
        help=f"Model to evaluate. Options: {', '.join(ALL_MODELS)}",
        nargs="?",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Local manifest created by dataset_preparation.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path",
    )
    parser.add_argument("--params-b", type=float)
    parser.add_argument("--memory-gb", type=float)
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models and exit",
    )

    args = parser.parse_args()

    if args.list_models:
        print("Available cloud ASR models:")
        print("\nOpenAI (OPENAI_API_KEY required):")
        for m in OPENAI_ASR_MODELS:
            print(f"  - {m}")
        print("\nGemini (GEMINI_API_KEY required):")
        for m in GEMINI_ASR_MODELS:
            print(f"  - {m}")
        return

    if not args.model:
        parser.error("model argument is required (use --list-models to see options)")

    if args.model not in ALL_MODELS:
        parser.error(
            f"Unknown model '{args.model}'. Use --list-models to see available options."
        )

    output_path = Path(args.output) if args.output else None

    t_start = time.time()
    model = load_model(args.model)
    result = evaluate_language(
        model=model,
        model_name=args.model,
        manifest_path=args.manifest,
    )

    elapsed = time.time() - t_start
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))

    results = {
        "model": args.model,
        "params_b": args.params_b,
        "memory_gb": args.memory_gb,
        "cloud": True,
        "hardware": hardware_string("cloud"),
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "benchmarks": {
            "fleurs_ca": {
                "wer": round(result.wer, 4),
                "cer": round(result.cer, 4),
                "rtf": None,
                "n": result.num_samples,
            }
        },
    }

    if output_path:
        write_json(output_path, results)
        print(f"\nResults saved to: {output_path}")

    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Model      : {args.model}")
    print(f"  WER        : {result.wer:.2%}")
    print(f"  CER        : {result.cer:.2%}")
    print(f"  RTF        : {result.avg_rtf:.3f}")
    print(f"  Samples    : {result.num_samples}")
    print(f"  Total time : {elapsed_str}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
