#!/usr/bin/env bash
uv run python model.py --model "bartowski/Llama-3.2-3B-Instruct-GGUF:Q8_0" "$@"
