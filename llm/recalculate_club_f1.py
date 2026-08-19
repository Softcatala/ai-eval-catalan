"""
Recalculate CLUB QA token F1 for configured LLM result files.

Runs each model's CLUB benchmark into a temporary JSON file, then merges only
benchmarks.club_qa back into the model's existing result JSON. A failing model
is reported and does not stop the remaining models.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from llamaserver import expected_gguf_filename
from run_evals import MODELS, SCRIPT_DIR, _llama_server_url_from_env


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _run_streamed(
    cmd: list[str], cwd: Path, env: dict, log_path: Path, timeout: int | None
) -> int | str:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        try:
            output, _ = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            output, _ = process.communicate()
            partial = exc.output or ""
            if isinstance(partial, bytes):
                partial = partial.decode(errors="replace")
            output = partial + (output or "")
            if output:
                print(output, end="")
                log.write(output)
            print(f"[ERROR] timed out after {timeout}s")
            log.write(f"[ERROR] timed out after {timeout}s\n")
            return "timeout"
        if output:
            print(output, end="")
            log.write(output)
        return process.returncode


def _format_command(cmd: list[str]) -> str:
    redacted = []
    redact_next = False
    for part in cmd:
        if redact_next:
            redacted.append("[REDACTED]")
            redact_next = False
            continue
        redacted.append(part)
        if part == "--api-key":
            redact_next = True
    return " ".join(redacted)


def _server_model_ids(server_url: str | None) -> set[str]:
    if not server_url:
        return set()
    url = server_url.rstrip("/")
    try:
        with urllib.request.urlopen(f"{url}/models", timeout=10) as resp:
            payload = json.loads(resp.read())
    except Exception as exc:
        print(f"[warn] could not query {url}/models: {exc}", flush=True)
        return set()
    return {item["id"] for item in payload.get("data", []) if "id" in item}


def _gguf_spec(model: dict) -> str | None:
    model_args = model.get("args", [])
    for i, arg in enumerate(model_args):
        if arg == "--model" and i + 1 < len(model_args):
            spec = model_args[i + 1]
            return None if spec in {"gemini", "openai", "claude"} else spec
    return None


def _request_model_id(model: dict, available_ids: set[str]) -> str | None:
    configured = model.get("llama_server_model")
    if configured and configured in available_ids:
        return configured

    spec = _gguf_spec(model)
    if not spec:
        return configured

    if spec in available_ids:
        return spec

    stem = Path(expected_gguf_filename(spec)).stem
    if stem in available_ids:
        return stem

    return configured


def _build_command(model: dict, args, output: Path, request_model: str | None) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        "model.py",
        *model["args"],
        "--output",
        str(output),
        "--n-samples",
        str(args.n_samples),
        "--benchmarks",
        "club",
    ]

    if args.llama_server_url and not model.get("cloud"):
        cmd += ["--llama-server-url", args.llama_server_url.rstrip("/")]
        llama_server_model = args.llama_server_model or request_model
        if llama_server_model:
            cmd += ["--llama-server-model", llama_server_model]

    if model.get("params_b") is not None:
        cmd += ["--params-b", str(model["params_b"])]

    cmd += ["--display-name", model["display_name"]]

    if model.get("cloud"):
        cmd += ["--cloud"]

    if model.get("quantized_analysis_only"):
        cmd += ["--quantized-analysis"]

    return cmd


def _missing_requirement(model: dict) -> str | None:
    if model.get("needs_api_key") and not os.environ.get("GOOGLE_API_KEY"):
        return "GOOGLE_API_KEY env var required but not set"
    if model.get("needs_openai_api_key") and not os.environ.get("OPENAI_API_KEY"):
        return "OPENAI_API_KEY env var required but not set"
    if model.get("needs_bedrock_token") and not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        return "AWS_BEARER_TOKEN_BEDROCK env var required but not set"
    if model.get("needs_openrouter_api_key") and not os.environ.get("OPENROUTER_API_KEY"):
        return "OPENROUTER_API_KEY env var required but not set"
    return None


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="Recalculate CLUB QA F1 results")
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument(
        "--models",
        nargs="+",
        help="Optional display-name subset from run_evals.MODELS",
    )
    parser.add_argument(
        "--llama-server-url",
        "--server-url",
        dest="llama_server_url",
        default=_llama_server_url_from_env(),
        help="OpenAI-compatible llama-server URL for local GGUF models.",
    )
    parser.add_argument(
        "--llama-server-model",
        "--server-model",
        dest="llama_server_model",
        default=None,
        help="Request model id. Only valid with one selected local model.",
    )
    parser.add_argument(
        "--model-timeout-seconds",
        type=int,
        default=0,
        help="Per-model timeout in seconds. Zero disables timeouts.",
    )
    args = parser.parse_args()

    selected = set(args.models) if args.models else None
    if selected:
        known = {model["display_name"] for model in MODELS}
        unknown = sorted(selected - known)
        if unknown:
            parser.error(f"unknown model display name(s): {', '.join(unknown)}")
        models = [model for model in MODELS if model["display_name"] in selected]
    else:
        models = MODELS

    local_models = [model for model in models if not model.get("cloud")]
    if args.llama_server_model and len(local_models) != 1:
        parser.error("--llama-server-model requires exactly one selected local GGUF model")

    updated: list[str] = []
    skipped: list[tuple[str, str]] = []
    failed: list[tuple[str, int | str]] = []
    available_ids = _server_model_ids(args.llama_server_url)

    tmpdir = Path(tempfile.mkdtemp(prefix="club-f1-"))
    print(f"[info] Temporary outputs: {tmpdir}")

    for model in models:
        name = model["display_name"]
        output_path = SCRIPT_DIR / model["output"]

        if not output_path.exists():
            skipped.append((name, f"{output_path} does not exist"))
            print(f"[SKIP] {name} - {output_path} does not exist")
            continue

        missing = _missing_requirement(model)
        if missing:
            skipped.append((name, missing))
            print(f"[SKIP] {name} - {missing}")
            continue

        request_model = None
        if not model.get("cloud") and args.llama_server_url:
            request_model = args.llama_server_model or _request_model_id(model, available_ids)
            spec = _gguf_spec(model)
            if spec and not request_model:
                reason = f"model is not available from {args.llama_server_url}"
                skipped.append((name, reason))
                print(f"[SKIP] {name} - {reason}")
                continue

        tmp_output = tmpdir / f"{name}.club.json"
        cmd = _build_command(model, args, tmp_output, request_model)
        print(f"\n[RUN] {name}: {_format_command(cmd)}\n{'=' * 60}")

        run_env = os.environ.copy()
        if model.get("needs_bedrock_token"):
            run_env["OPENAI_API_KEY"] = run_env["AWS_BEARER_TOKEN_BEDROCK"]
        if model.get("needs_openrouter_api_key"):
            cmd += ["--api-key", run_env["OPENROUTER_API_KEY"]]

        log_path = tmpdir / f"{name}.log"
        timeout = args.model_timeout_seconds or None
        returncode = _run_streamed(cmd, SCRIPT_DIR, run_env, log_path, timeout)
        if returncode != 0:
            failed.append((name, f"{returncode}; log {log_path}"))
            print(f"[ERROR] {name} failed: {returncode}; log {log_path}")
            continue

        try:
            fresh = _load_json(tmp_output)
            club_qa = fresh["benchmarks"]["club_qa"]
            existing = _load_json(output_path)
            existing.setdefault("benchmarks", {})["club_qa"] = club_qa
            _write_json(output_path, existing)
        except Exception as exc:
            failed.append((name, str(exc)))
            print(f"[ERROR] {name} failed to merge result: {exc}")
            continue

        updated.append(name)
        print(f"[DONE] {name} -> {output_path} club_qa={club_qa}")

    print("\nSUMMARY")
    print(f"  updated: {len(updated)}")
    print(f"  skipped: {len(skipped)}")
    print(f"  failed : {len(failed)}")
    if skipped:
        for name, reason in skipped:
            print(f"  skip   {name}: {reason}")
    if failed:
        for name, reason in failed:
            print(f"  fail   {name}: {reason}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
