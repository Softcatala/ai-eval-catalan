"""Evaluate discovered decision models, skipping completed matching results."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

try:
    from .models_config import MODELS
    from .model import (
        SCRIPT_DIR,
        add_evaluation_arguments,
        discover_models,
        validate_arguments,
    )
except ImportError:
    from models_config import MODELS
    from model import (
        SCRIPT_DIR,
        add_evaluation_arguments,
        discover_models,
        validate_arguments,
    )


def output_path(directory, model):
    for configured in MODELS:
        if configured["model"] == model:
            return directory / Path(configured["output"]).name
    slug = re.sub(r"[^A-Za-z0-9._-]", "_", model)
    return directory / f"{slug}.json"


def completed(path, args, model):
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        result.get("model") == model
        and result.get("provider", "systemone") == args.provider
        and result.get("locale") == args.locale
        and result.get("labels") == args.labels
        and result.get("seed") == args.seed
        and result.get("shuffled_options") == args.shuffle_options
        and result.get("requested_n_samples") == args.limit
        and result.get("data_source")
        == (str(args.data.resolve()) if args.data else "MASSIVE 1.1")
        and "accuracy" in result
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_evaluation_arguments(parser)
    parser.add_argument(
        "--models",
        nargs="+",
        help="Exact server model IDs; default: discover all decision models",
    )
    parser.add_argument(
        "--server-model", help="Override request ID for one selected model"
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "evals")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    validate_arguments(parser, args)
    if args.server_model and (not args.models or len(args.models) != 1):
        parser.error("--server-model requires exactly one model in --models")
    try:
        models = (
            list(dict.fromkeys(args.models))
            if args.models
            else [m["model"] for m in MODELS if m.get("provider") == "openai"]
            if args.provider == "openai"
            else discover_models(args)
        )
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f"Cannot discover models: {error}\n")
    if not models:
        parser.exit(1, "No decision models found at /v1/models.\n")
    outputs = [output_path(args.output_dir, model) for model in models]
    if len(set(outputs)) != len(outputs):
        parser.error("Selected model IDs produce duplicate output filenames")
    failures = []
    for model in models:
        output = output_path(args.output_dir, model)
        if (
            not args.overwrite
            and not args.server_model
            and completed(output, args, model)
        ):
            print(f"[SKIP] {model}: {output}", flush=True)
            continue
        command = [
            sys.executable,
            "-u",
            str(SCRIPT_DIR / "model.py"),
            "--model",
            model,
            "--output",
            str(output),
        ]
        for option in (
            "server_url",
            "provider",
            "limit",
            "locale",
            "labels",
            "seed",
            "cache",
            "timeout",
            "data",
            "server_model",
        ):
            value = getattr(args, option)
            if value is not None:
                command += ["--" + option.replace("_", "-"), str(value)]
        if args.shuffle_options:
            command.append("--shuffle-options")
        print(f"[RUN] {model}", flush=True)
        result = subprocess.run(command, stdin=subprocess.DEVNULL)
        if result.returncode:
            failures.append(model)
            print(f"[ERROR] {model}: exit code {result.returncode}", flush=True)
        else:
            print(f"[DONE] {model}: {output}", flush=True)
    if failures:
        parser.exit(1, f"Failed models: {', '.join(failures)}\n")


if __name__ == "__main__":
    main()
