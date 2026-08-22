"""Shared COMET configuration for FLORES translation evaluations."""

COMET_CHECKPOINT = "Unbabel/wmt22-comet-da"
LEGACY_TRANSLATION_METRICS = (
    "bleu,none",
    "bleu_stderr,none",
    "ter,none",
    "ter_stderr,none",
    "chrf,none",
    "chrf_stderr,none",
)


def drop_legacy_translation_metrics(result: dict) -> None:
    for metric in LEGACY_TRANSLATION_METRICS:
        result.pop(metric, None)
