#!/usr/bin/env python3
"""
WER Evaluation Script for ASR Models
Evaluates Word Error Rate on FLEURS dataset for Catalan and English.
Supports: Omnilingual ASR and OpenAI Whisper models.

Usage:
    python evaluate_wer.py whisper-small whisper-large-v3 --num_samples 500
    python evaluate_wer.py omniASR_CTC_300M omniASR_CTC_1B --device cuda
    python evaluate_wer.py whisper-small omniASR_CTC_300M --output results.csv
    python evaluate_wer.py --list-models
"""

import argparse
import time
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
import torch
import torchaudio
from datasets import load_dataset
from jiwer import wer, cer
from tqdm import tqdm


@dataclass
class EvalResult:
    """Stores evaluation results for a language."""

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
    #    "en": {
    #        "name": "English",
    #        "omni_lang": "eng_Latn",
    #        "whisper_lang": "english",
    #        "fleurs_locale": "en_us",
    #    },
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
    "whisper-large-v2",
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
    """Protocol for ASR models."""

    def transcribe(
        self, waveform: torch.Tensor, sample_rate: int, lang: str
    ) -> str: ...


class OmnilangualASRWrapper:
    """Wrapper for Omnilingual ASR models."""

    def __init__(self, model_name: str, device: str):
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

        self.pipeline = ASRInferencePipeline(model_card=model_name, device=device)
        self.model_name = model_name

    def transcribe(self, waveform: torch.Tensor, sample_rate: int, lang: str) -> str:
        audio_data = [{"waveform": waveform, "sample_rate": sample_rate}]
        result = self.pipeline.transcribe(audio_data, lang=[lang], batch_size=1)
        return result[0] if result else ""


class WhisperWrapper:
    """Wrapper for OpenAI Whisper models via transformers."""

    def __init__(self, model_name: str, device: str):
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        # Map short names to HuggingFace model IDs
        if "aina" not in model_name:
            model_id = f"openai/{model_name.replace('whisper-', 'whisper-')}"
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
        # Whisper expects numpy array
        audio = waveform.numpy()

        result = self.pipe(
            {"array": audio, "sampling_rate": sample_rate},
            generate_kwargs={"language": lang},
        )
        return result["text"] if result else ""


class VibeVoiceWrapper:
    """Wrapper for Microsoft VibeVoice-ASR (custom library required)."""

    def __init__(self, model_name: str, device: str):
        from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
        from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

        self.processor = VibeVoiceASRProcessor.from_pretrained(
            model_name,
            language_model_pretrained_name="Qwen/Qwen2.5-7B",
        )
        self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            trust_remote_code=True,
        ).to(device)
        self.model.eval()
        self.device = device
        self.model_name = model_name

    def transcribe(self, waveform: torch.Tensor, sample_rate: int, lang: str) -> str:
        # Processor expects audio at 24kHz; resample if needed
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
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

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
        eos_pos = (generated_ids == self.processor.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_pos) > 0:
            generated_ids = generated_ids[: eos_pos[0] + 1]
        raw_text = self.processor.decode(generated_ids, skip_special_tokens=True)

        PLACEHOLDERS = {"[Silence]", "[Unintelligible Speech]", "[noise]", "[music]"}
        try:
            segments = self.processor.post_process_transcription(raw_text)
            if segments:
                texts = [seg.get("text", "") for seg in segments if seg.get("text", "") not in PLACEHOLDERS]
                return " ".join(texts).strip()
        except Exception:
            pass
        return raw_text


class Gemma4Wrapper:
    """Wrapper for Gemma 4 multimodal models with audio/ASR support."""

    MAX_AUDIO_DURATION = 30.0  # Gemma 4 audio limit in seconds

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
        import tempfile

        import soundfile as sf

        # Gemma 4 expects a file path; write waveform to a temp WAV file
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
            import os

            os.unlink(tmp_path)


def load_model(model_name: str, device: str) -> ASRModel:
    """Load the appropriate ASR model based on name."""
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
    """
    Normalize text for WER calculation.
    - Lowercase
    - Remove punctuation
    - Normalize whitespace
    """
    import re
    import unicodedata

    # Lowercase
    text = text.lower()

    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)

    # Remove punctuation (keep letters, numbers, spaces)
    text = re.sub(r"[^\w\s]", "", text)

    # Normalize whitespace
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
    """
    Evaluate WER for a specific language using FLEURS dataset.

    Args:
        model: ASR model wrapper
        model_name: Name of the model (for determining lang code format)
        lang_code: Language code (e.g., "ca", "en")
        num_samples: Number of samples to evaluate
        max_duration: Maximum audio duration in seconds
        warmup: Number of initial samples to exclude from latency measurements

    Returns:
        EvalResult with WER, CER, and timing information
    """
    lang_config = LANGUAGE_CONFIG[lang_code]
    locale = lang_config["fleurs_locale"]

    # Determine which language code format to use
    if model_name in OMNILINGUAL_MODELS:
        model_lang = lang_config["omni_lang"]
    else:
        model_lang = lang_config["whisper_lang"]

    print(f"\n{'='*60}")
    print(f"Evaluating {lang_config['name']} ({lang_code}) on FLEURS")
    print(f"Model: {model_name} | Lang code: {model_lang}")
    print(f"{'='*60}")

    # Load FLEURS dataset in streaming mode
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
    rtfs = []  # Real-Time Factor per sample
    skipped = 0
    processed = 0
    start_time = time.time()

    # Create resampler once outside the loop (FLEURS is typically 48kHz)
    resampler = torchaudio.transforms.Resample(48000, 16000)

    with torch.no_grad():  # Disable gradient tracking for inference
        for sample in tqdm(
            dataset, desc=f"Processing {lang_config['name']}", total=num_samples
        ):
            # Stop after processing enough samples
            if processed >= num_samples:
                break

            try:
                # Get reference text (FLEURS uses "transcription" field)
                reference = sample["transcription"]

                # Skip if audio is too long (model limitation)
                audio_array = sample["audio"]["array"]
                sample_rate = sample["audio"]["sampling_rate"]
                duration = len(audio_array) / sample_rate

                if duration > max_duration:
                    skipped += 1
                    continue

                # Prepare audio data
                waveform = torch.tensor(audio_array, dtype=torch.float32)

                # Resample to 16kHz if needed
                if sample_rate != 16000:
                    # Recreate resampler only if sample rate differs from expected
                    if sample_rate != 48000:
                        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                    waveform = resampler(waveform.unsqueeze(0)).squeeze(0)
                    sample_rate = 16000

                # Transcribe and measure time
                inference_start = time.perf_counter()
                hypothesis = model.transcribe(waveform, sample_rate, model_lang)
                inference_end = time.perf_counter()

                # Calculate RTF (skip warmup samples)
                if processed >= warmup:
                    rtf = (inference_end - inference_start) / duration
                    rtfs.append(rtf)

                # Normalize texts
                ref_normalized = normalize_text(reference)
                hyp_normalized = normalize_text(hypothesis)

                if ref_normalized:
                    references.append(ref_normalized)
                    hypotheses.append(hyp_normalized)

                processed += 1

                # Cleanup
                del waveform, audio_array

                # Periodically clear CUDA cache
                if processed % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"\nError processing sample: {e}")
                skipped += 1
                continue

    total_time = time.time() - start_time

    # Calculate metrics
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
        f"  RTF: {result.avg_rtf:.3f} ({1/result.avg_rtf:.1f}x real-time)"
        if result.avg_rtf > 0
        else "  RTF: N/A"
    )

    return result


def print_summary(all_results: list[tuple[str, EvalResult]]):
    """Print summary table of all results."""
    print(f"\n{'='*80}")
    print(f"SUMMARY | Dataset: FLEURS")
    print(f"{'='*80}")
    print(
        f"{'Model':<25} {'Language':<12} {'Samples':<10} {'WER':<10} {'CER':<10} {'RTF':<10}"
    )
    print("-" * 80)

    for model_name, r in all_results:
        print(
            f"{model_name:<25} {r.language:<12} {r.num_samples:<10} {r.wer:<10.2%} {r.cer:<10.2%} {r.avg_rtf:<10.3f}"
        )

    print("-" * 80)
    print(f"\nRTF = Real-Time Factor (< 1 = faster than real-time)")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ASR models (Omnilingual ASR / Whisper) on FLEURS dataset"
    )
    parser.add_argument(
        "models",
        type=str,
        nargs="*",
        help=f"Models to evaluate. Options: {', '.join(ALL_MODELS)}",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run inference on",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to evaluate per language",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file for results",
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

    if not args.models:
        print("Error: No models specified")
        print(f"Usage: python evaluate_wer.py MODEL1 [MODEL2 ...]")
        print("Use --list-models to see all options")
        return

    # Validate model names
    for model_name in args.models:
        if model_name not in ALL_MODELS:
            print(f"Error: Unknown model '{model_name}'")
            print(f"Available models: {', '.join(ALL_MODELS)}")
            print("Use --list-models to see all options")
            return

    languages = ["ca"]  # Always evaluate both

    print(f"ASR Model WER Evaluation")
    print(f"Models: {', '.join(args.models)}")
    print(f"Device: {args.device}")
    print(f"Dataset: FLEURS")
    print(f"Languages: Catalan, English")
    print(f"Samples per language: {args.num_samples}")

    all_results = []  # (model_name, EvalResult)

    for model_name in args.models:
        print(f"\n{'#'*70}")
        print(f"# Loading model: {model_name}")
        print(f"{'#'*70}")
        model = load_model(model_name, args.device)
        print("Model loaded successfully!")

        max_duration = Gemma4Wrapper.MAX_AUDIO_DURATION if model_name in GEMMA_MODELS else 40.0
        for lang_code in languages:
            result = evaluate_language(
                model=model,
                model_name=model_name,
                lang_code=lang_code,
                num_samples=args.num_samples,
                max_duration=max_duration,
            )
            all_results.append((model_name, result))

    # Print summary
    print_summary(all_results)

    # Save to CSV if requested
    if args.output:
        import csv

        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "language", "num_samples", "wer", "cer", "rtf"])
            for model_name, r in all_results:
                writer.writerow(
                    [
                        model_name,
                        r.language,
                        r.num_samples,
                        f"{r.wer:.4f}",
                        f"{r.cer:.4f}",
                        f"{r.avg_rtf:.4f}",
                    ]
                )
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
