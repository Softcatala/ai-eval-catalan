"""Explicit result filenames for decision models."""

MODELS = [
    {
        "model": "gpt-6-luna",
        "provider": "openai",
        "output": "evals/gpt_6_luna_decisions.json",
    },
    {"model": "Bespoke-Nimble-9B-v3-GGUF", "output": "evals/nimble_9b.json"},
    {"model": "Clef-Flash-GGUF", "output": "evals/clef_flash.json"},
    {"model": "Clef-GGUF", "output": "evals/clef.json"},
    {"model": "Julia-1-GGUF", "output": "evals/julia_1.json"},
    {"model": "Kev-4B-GGUF", "output": "evals/kev_4b.json"},
    {"model": "Kev-9B-GGUF", "output": "evals/kev_9b.json"},
    {"model": "Laya-GGUF", "output": "evals/laya.json"},
    {"model": "OpenJev-GGUF", "output": "evals/openjev.json"},
    {"model": "lev-GGUF", "output": "evals/lev.json"},
    {
        "model": "owao/surogate-rune-26b-a4b-GGUF:Q8_0",
        "output": "evals/rune_26b_a4b_v3_q8_0.json",
    },
]
