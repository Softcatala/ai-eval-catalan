#!/usr/bin/env python3
"""
WER Evaluation Script for a single ASR model.
Evaluates Word Error Rate on FLEURS dataset for Catalan.
Writes results to a JSON file with the same structure as llm/model.py.

Usage:
    python hf-eval.py whisper-small --output evals/results_whisper_small.json
    python hf-eval.py omniASR_CTC_300M --device cuda --output evals/results_omni_ctc_300m.json
    python hf-eval.py --list-models
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

import numpy as np
import torch
import torchaudio
from datasets import load_dataset
from jiwer import wer, cer
from tqdm import tqdm


@dataclass
class EvalResult:
    language: str
    num_samples: int
    wer: float
    cer: float
    total_time: float
    avg_rtf: float  # Real-Time Factor (processing_time / audio_duration)


# Language configuration: FLEURS locale -> model lang codes
LANGUAGE_CONFIG = {
    "ca": {
        "name": "Catalan",
        "omni_lang": "cat_Latn",
        "whisper_lang": "catalan",
        "fleurs_locale": "ca_es",
    },
}

# Model configurations
OMNILINGUAL_MODELS = [
    "omniASR_CTC_300M",
    "omniASR_CTC_1B",
    "omniASR_CTC_3B",
    "omniASR_CTC_7B",
    "omniASR_LLM_300M",
    "omniASR_LLM_1B",
    "omniASR_LLM_3B",
    "omniASR_LLM_7B",
]

WHISPER_MODELS = [
    "whisper-tiny",
    "whisper-base",
    "whisper-small",
    "whisper-medium",
    "whisper-large",
    "whisper-large-v3",
    "whisper-large-v3-turbo",
    "projecte-aina/whisper-large-v3-ca-3catparla",
]

VIBEVOICE_MODELS = [
    "microsoft/VibeVoice-ASR",
]

GEMMA_MODELS = [
    "gemma-4-E4B",
    "gemma-4-E2B",
]

ALL_MODELS = OMNILINGUAL_MODELS + WHISPER_MODELS + VIBEVOICE_MODELS + GEMMA_MODELS


class ASRModel(Protocol):
    def transcribe(
        self, waveform: torch.Tensor, sample_rate: int, lang: str
    ) -> str: ...


class OmnilangualASRWrapper:
    def __init__(self, model_name: str, device: str):
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

        self.pipeline = ASRInferencePipeline(model_card=model_name, device=device)
        self.model_name = model_name

    def transcribe(self, waveform: torch.Tensor, sample_rate: int, lang: str) -> str:
        audio_data = [{"waveform": waveform, "sample_rate": sample_rate}]
        result = self.pipeline.transcribe(audio_data, lang=[lang], batch_size=1)
        return result[0] if result else ""


class WhisperWrapper:
    def __init__(self, model_name: str, device: str):
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        if "aina" not in model_name:
            model_id = f"openai/{model_name}"
        else:
            model_id = model_name

        self.device = device
        torch_dtype = torch.float16 if device == "cuda" else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )
        self.model_name = model_name

    def transcribe(self, waveform: torch.Tensor, sample_rate: int, lang: str) -> str:
        audio = waveform.numpy()
        result = self.pipe(
            {"array": audio, "sampling_rate": sample_rate},
            generate_kwargs={"language": lang},
        )
        return result["text"] if result else ""


class VibeVoiceWrapper:
    def __init__(self, model_name: str, device: str):
        from vibevoice.modular.modeling_vibevoice_asr import (
            VibeVoiceASRForConditionalGeneration,
        )
        from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

        self.processor = VibeVoiceASRProcessor.from_pretrained(
            model_name,
            language_model_pretrained_name="Qwen/Qwen2.5-7B",
        )
        self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            trust_remote_code=True,
        ).to(device)
        self.model.eval()
        self.device = device
        self.model_name = model_name

    def transcribe(self, waveform: torch.Tensor, sample_rate: int, lang: str) -> str:
        target_sr = self.processor.target_sample_rate
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform.unsqueeze(0)).squeeze(0)

        inputs = self.processor(
            audio=[waveform.numpy()],
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
            context_info="The audio is in Catalan language.",
        )
        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                pad_token_id=self.processor.pad_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                do_sample=False,
            )

        input_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, input_length:]
        eos_pos = (generated_ids == self.processor.tokenizer.eos_token_id).nonzero(
            as_tuple=True
        )[0]
        if len(eos_pos) > 0:
            generated_ids = generated_ids[: eos_pos[0] + 1]
        raw_text = self.processor.decode(generated_ids, skip_special_tokens=True)

        PLACEHOLDERS = {"[Silence]", "[Unintelligible Speech]", "[noise]", "[music]"}
        try:
            segments = self.processor.post_process_transcription(raw_text)
            if segments:
                texts = [
                    seg.get("text", "")
                    for seg in segments
                    if seg.get("text", "") not in PLACEHOLDERS
                ]
                return " ".join(texts).strip()
        except Exception:
            pass
        return raw_text


class Gemma4Wrapper:
    MAX_AUDIO_DURATION = 30.0

    def __init__(self, model_name: str, device: str):
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        model_id = f"google/{model_name}-it"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_id,
            dtype="auto",
            device_map="auto",
        )
        self.model_name = model_name

    def transcribe(self, waveform: torch.Tensor, sample_rate: int, lang: str) -> str:
        import os
        import tempfile

        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            sf.write(tmp_path, waveform.numpy(), sample_rate)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": tmp_path},
                        {
                            "type": "text",
                            "text": (
                                f"Transcribe the following {lang} speech segment verbatim into {lang}. "
                                "Do not translate. Do not paraphrase. Do not add any commentary.\n"
                                "Formatting rules:\n"
                                "* Output only the transcription, nothing else, with no newlines.\n"
                                "* Do not add punctuation unless it was clearly spoken.\n"
                                "* Write numbers as digits (e.g. 3 not three, 1.7 not one point seven)."
                            ),
                        },
                    ],
                }
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            ).to(self.model.device)
            input_len = inputs["input_ids"].shape[-1]

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=1.0,
                repetition_penalty=1.0,
            )
            response = self.processor.decode(
                outputs[0][input_len:], skip_special_tokens=False
            )
            parsed = self.processor.parse_response(response)
            if isinstance(parsed, dict):
                return parsed.get("text", parsed.get("transcription", str(parsed)))
            return parsed
        finally:
            os.unlink(tmp_path)


def load_model(model_name: str, device: str) -> ASRModel:
    if model_name in OMNILINGUAL_MODELS:
        print(f"Loading Omnilingual ASR model: {model_name}")
        return OmnilangualASRWrapper(model_name, device)
    elif model_name in WHISPER_MODELS:
        print(f"Loading Whisper model: {model_name}")
        return WhisperWrapper(model_name, device)
    elif model_name in VIBEVOICE_MODELS:
        print(f"Loading VibeVoice model: {model_name}")
        return VibeVoiceWrapper(model_name, device)
    elif model_name in GEMMA_MODELS:
        print(f"Loading Gemma 4 model: {model_name}")
        return Gemma4Wrapper(model_name, device)
    else:
        raise ValueError(f"Unknown model: {model_name}. Available: {ALL_MODELS}")


def normalize_text(text: str) -> str:
    import re
    import unicodedata

    text = text.lower()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def evaluate_language(
    model: ASRModel,
    model_name: str,
    lang_code: str,
    num_samples: int,
    max_duration: float = 40.0,
    warmup: int = 3,
) -> EvalResult:
    lang_config = LANGUAGE_CONFIG[lang_code]
    locale = lang_config["fleurs_locale"]

    if model_name in OMNILINGUAL_MODELS:
        model_lang = lang_config["omni_lang"]
    else:
        model_lang = lang_config["whisper_lang"]

    print(f"\n{'=' * 60}")
    print(f"Evaluating {lang_config['name']} ({lang_code}) on FLEURS")
    print(f"Model: {model_name} | Lang code: {model_lang}")
    print(f"{'=' * 60}")

    print(f"Loading FLEURS dataset for {lang_config['name']} (streaming)...")
    dataset = load_dataset(
        "google/fleurs",
        locale,
        split="test",
        streaming=True,
        trust_remote_code=True,
    )

    print(f"Evaluating on {num_samples} samples...")

    references = []
    hypotheses = []
    rtfs = []
    skipped = 0
    processed = 0
    start_time = time.time()

    resampler = torchaudio.transforms.Resample(48000, 16000)

    with torch.no_grad():
        for sample in tqdm(
            dataset, desc=f"Processing {lang_config['name']}", total=num_samples
        ):
            if processed >= num_samples:
                break

            try:
                reference = sample["transcription"]
                audio_array = sample["audio"]["array"]
                sample_rate = sample["audio"]["sampling_rate"]
                duration = len(audio_array) / sample_rate

                if duration > max_duration:
                    skipped += 1
                    continue

                waveform = torch.tensor(audio_array, dtype=torch.float32)

                if sample_rate != 16000:
                    if sample_rate != 48000:
                        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                    waveform = resampler(waveform.unsqueeze(0)).squeeze(0)
                    sample_rate = 16000

                inference_start = time.perf_counter()
                hypothesis = model.transcribe(waveform, sample_rate, model_lang)
                inference_end = time.perf_counter()

                if processed >= warmup:
                    rtf = (inference_end - inference_start) / duration
                    rtfs.append(rtf)

                ref_normalized = normalize_text(reference)
                hyp_normalized = normalize_text(hypothesis)

                if ref_normalized:
                    references.append(ref_normalized)
                    hypotheses.append(hyp_normalized)

                processed += 1

                del waveform, audio_array

                if processed % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

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
        description="Evaluate a single ASR model on FLEURS and write a JSON results file"
    )
    parser.add_argument(
        "model",
        type=str,
        help=f"Model to evaluate. Options: {', '.join(ALL_MODELS)}",
        nargs="?",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run inference on (default: cpu)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=200,
        help="Number of samples to evaluate (default: 200)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: evals/results_<model>.json)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models and exit",
    )

    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        print("\nOmnilingual ASR:")
        for m in OMNILINGUAL_MODELS:
            print(f"  - {m}")
        print("\nWhisper:")
        for m in WHISPER_MODELS:
            print(f"  - {m}")
        print("\nVibeVoice (custom library required):")
        for m in VIBEVOICE_MODELS:
            print(f"  - {m}")
        print("\nGemma 4 (audio, max 30s):")
        for m in GEMMA_MODELS:
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
    model = load_model(args.model, args.device)

    max_duration = (
        Gemma4Wrapper.MAX_AUDIO_DURATION if args.model in GEMMA_MODELS else 40.0
    )
    result = evaluate_language(
        model=model,
        model_name=args.model,
        lang_code="ca",
        num_samples=args.num_samples,
        max_duration=max_duration,
    )

    elapsed = time.time() - t_start
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))

    results = {
        "model": args.model,
        "benchmarks": {
            "fleurs_ca": {
                "wer": round(result.wer, 4),
                "cer": round(result.cer, 4),
                "rtf": round(result.avg_rtf, 4),
                "n": result.num_samples,
            }
        },
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
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
