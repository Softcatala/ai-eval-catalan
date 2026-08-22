"""Crash-safe full evaluation runner for the BLEU-to-COMET migration."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from comet_config import COMET_CHECKPOINT, LEGACY_TRANSLATION_METRICS
from run_evals import DEFAULT_LOCAL_SERVER_URL, MODELS

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
STATE_PATH = Path(os.environ.get("COMET_STATE_PATH", ROOT_DIR / "comet_eval_state.json"))
LOG_DIR = SCRIPT_DIR / "comet_logs"
PROGRESS_INTERVAL_SECONDS = 600
EXPECTED_BENCHMARKS = {"sts_ca", "catcola", "club_qa", "casum", "flores", "ifeval"}
FLORES_TASKS = {
    "catalan_bench_flores_en-ca",
    "catalan_bench_flores_ca-en",
    "catalan_bench_flores_es-ca",
    "catalan_bench_flores_ca-es",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def output_path(model: dict) -> Path:
    return SCRIPT_DIR / model["output"]


def validate_result(path: Path, n_samples: int = 400) -> tuple[bool, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"invalid JSON: {exc}"
    benchmarks = data.get("benchmarks")
    if not isinstance(benchmarks, dict):
        return False, "missing benchmarks"
    missing = EXPECTED_BENCHMARKS - benchmarks.keys()
    if missing:
        return False, f"missing benchmarks: {', '.join(sorted(missing))}"
    for name in EXPECTED_BENCHMARKS:
        result = benchmarks[name]
        if not isinstance(result, dict) or "error" in result:
            return False, f"invalid benchmark: {name}"
    flores = benchmarks["flores"]
    missing = FLORES_TASKS - flores.keys()
    if missing:
        return False, f"missing FLORES tasks: {', '.join(sorted(missing))}"
    for task in FLORES_TASKS:
        result = flores[task]
        comet = result.get("comet,none")
        if not isinstance(comet, (int, float)) or not math.isfinite(comet):
            return False, f"invalid COMET score: {task}"
        if result.get("n") != n_samples:
            return False, f"invalid sample count: {task}"
        if legacy := set(result).intersection(LEGACY_TRANSLATION_METRICS):
            return False, f"obsolete translation metric: {task} ({', '.join(sorted(legacy))})"
    return True, "ok"


def merge_flores(temporary: Path, destination: Path, n_samples: int) -> None:
    fresh = json.loads(temporary.read_text(encoding="utf-8"))
    existing = json.loads(destination.read_text(encoding="utf-8"))
    flores = fresh.get("benchmarks", {}).get("flores")
    if not isinstance(flores, dict):
        raise RuntimeError("fresh result is missing FLORES")
    candidate = dict(existing)
    candidate["benchmarks"] = dict(existing.get("benchmarks", {}))
    candidate["benchmarks"]["flores"] = flores
    candidate["flores_evaluated_at"] = fresh.get("evaluated_at", now())
    candidate["flores_comet_checkpoint"] = fresh.get(
        "flores_comet_checkpoint", COMET_CHECKPOINT
    )
    staging = destination.with_name(f".{destination.name}.merged.tmp")
    atomic_json(staging, candidate)
    valid, reason = validate_result(staging, n_samples)
    if not valid:
        staging.unlink(missing_ok=True)
        raise RuntimeError(reason)
    os.replace(staging, destination)
    temporary.unlink(missing_ok=True)


def initial_state(n_samples: int) -> dict:
    return {
        "version": 1,
        "created_at": now(),
        "updated_at": now(),
        "n_samples": n_samples,
        "models": {
            model["display_name"]: {
                "status": "pending",
                "attempts": 0,
                "started_at": None,
                "ended_at": None,
                "output": str(output_path(model).relative_to(ROOT_DIR)),
                "log": str((LOG_DIR / f"{model['display_name']}.log").relative_to(ROOT_DIR)),
                "last_error": None,
                "duration_seconds": None,
            }
            for model in MODELS
        },
    }


def load_state(n_samples: int) -> dict:
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if state.get("n_samples") != n_samples:
            raise RuntimeError("checkpoint sample count does not match --n-samples")
    else:
        state = initial_state(n_samples)
    known = state.setdefault("models", {})
    for model in MODELS:
        name = model["display_name"]
        known.setdefault(name, initial_state(n_samples)["models"][name])
        entry = known[name]
        valid, _ = validate_result(output_path(model), n_samples)
        if entry.get("status") == "complete" and valid:
            continue
        if valid:
            entry.update(status="complete", ended_at=entry.get("ended_at") or now(), last_error=None)
        elif entry.get("status") == "running":
            entry.update(status="pending", last_error="interrupted before completion")
        elif entry.get("status") == "complete":
            entry.update(status="pending", last_error="completed output failed validation")
        temporary = output_path(model).with_name(f".{output_path(model).name}.comet.tmp")
        if temporary.exists():
            temporary.unlink()
    state["updated_at"] = now()
    atomic_json(STATE_PATH, state)
    return state


def model_command(model: dict, temporary: Path, n_samples: int) -> tuple[list[str], dict]:
    env = os.environ.copy()
    conda_lib = "/opt/conda/lib"
    current_library_path = env.get("LD_LIBRARY_PATH")
    env["LD_LIBRARY_PATH"] = (
        f"{conda_lib}:{current_library_path}" if current_library_path else conda_lib
    )
    command = [
        sys.executable,
        "-u",
        "model.py",
        *model["args"],
        "--output",
        str(temporary.relative_to(SCRIPT_DIR)),
        "--n-samples",
        str(n_samples),
        "--benchmarks",
        "flores",
        "--display-name",
        model["display_name"],
    ]
    if not model.get("cloud"):
        command += ["--llama-server-url", DEFAULT_LOCAL_SERVER_URL]
        if model.get("llama_server_model"):
            command += ["--llama-server-model", model["llama_server_model"]]
    else:
        command.append("--cloud")
    if model.get("params_b") is not None:
        command += ["--params-b", str(model["params_b"])]
    if model.get("quantized_analysis_only"):
        command.append("--quantized-analysis")
    if model.get("needs_api_key"):
        api_key = env.get("GOOGLE_API_KEY") or env.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY or GEMINI_API_KEY is required")
        command += ["--api-key", api_key]
    if model.get("needs_openai_api_key") and not env.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required")
    if model.get("needs_bedrock_token"):
        token = env.get("AWS_BEARER_TOKEN_BEDROCK")
        if not token:
            raise RuntimeError("AWS_BEARER_TOKEN_BEDROCK is required")
        env["OPENAI_API_KEY"] = token
    return command, env


def ensure_local_model_available(model: dict) -> None:
    if model.get("cloud"):
        return
    request_model = model.get("llama_server_model")
    if not request_model:
        model_flag = model["args"].index("--model")
        request_model = model["args"][model_flag + 1]
    with urllib.request.urlopen(f"{DEFAULT_LOCAL_SERVER_URL}/models", timeout=10) as response:
        available = {item["id"] for item in json.load(response).get("data", [])}
    if request_model not in available:
        raise RuntimeError(f"model is not available on the shared server: {request_model}")


def progress(state: dict, started: float) -> str:
    entries = state["models"]
    complete = [name for name, item in entries.items() if item["status"] == "complete"]
    running = [name for name, item in entries.items() if item["status"] == "running"]
    failed = [name for name, item in entries.items() if item["status"] == "failed"]
    left = [name for name, item in entries.items() if item["status"] != "complete"]
    durations = [item["duration_seconds"] for item in entries.values() if item.get("duration_seconds")]
    eta = "unknown"
    if durations:
        eta = f"{sum(durations) / len(durations) * len(left) / 3600:.1f}h"
    return (
        f"[PROGRESS] {len(complete)}/{len(entries)} complete; {len(left)} left "
        f"({', '.join(left) or 'none'}); running={running or ['none']}; "
        f"failed={failed or ['none']}; elapsed={(time.time() - started) / 3600:.1f}h; ETA={eta}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="append", default=[], metavar="MODEL")
    parser.add_argument("--models", nargs="+", metavar="MODEL")
    parser.add_argument("--n-samples", type=int, default=400)
    args = parser.parse_args()
    state = load_state(args.n_samples)
    unknown = set(args.force) - {model["display_name"] for model in MODELS}
    if args.models:
        unknown |= set(args.models) - {model["display_name"] for model in MODELS}
    if unknown:
        parser.error(f"unknown model(s): {', '.join(sorted(unknown))}")
    for name in args.force:
        state["models"][name].update(status="pending", last_error="forced rerun")
    atomic_json(STATE_PATH, state)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    children: list[subprocess.Popen] = []
    active_names: set[str] = set()

    def stop(signum, _frame):
        for process in children:
            if process.poll() is None:
                process.terminate()
        for name in active_names:
            state["models"][name].update(
                status="pending", last_error=f"interrupted by signal {signum}"
            )
        state["updated_at"] = now()
        atomic_json(STATE_PATH, state)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print(progress(state, started), flush=True)

    pending_score = None

    def finish_score(job) -> None:
        if job is None:
            return
        process, model, temporary, examples, model_started = job
        name = model["display_name"]
        entry = state["models"][name]
        next_progress = time.time() + PROGRESS_INTERVAL_SECONDS
        while process.poll() is None:
            time.sleep(10)
            if time.time() >= next_progress:
                print(progress(state, started), flush=True)
                next_progress += PROGRESS_INTERVAL_SECONDS
        try:
            if process.returncode:
                raise RuntimeError(f"COMET scorer exited with code {process.returncode}")
            merge_flores(temporary, output_path(model), args.n_samples)
            entry.update(
                status="complete",
                ended_at=now(),
                duration_seconds=round(time.time() - model_started, 1),
                last_error=None,
            )
        except Exception as exc:
            entry.update(status="failed", ended_at=now(), last_error=str(exc))
            temporary.unlink(missing_ok=True)
            examples.unlink(missing_ok=True)
        active_names.discard(name)
        state["updated_at"] = now()
        atomic_json(STATE_PATH, state)
        print(progress(state, started), flush=True)

    try:
        selected = set(args.models) if args.models else None
        for model in MODELS:
            if selected and model["display_name"] not in selected:
                continue
            name = model["display_name"]
            entry = state["models"][name]
            if entry["status"] == "complete":
                continue
            if entry["status"] == "failed" and entry["attempts"] >= 2:
                continue
            destination = output_path(model)
            temporary = destination.with_name(f".{destination.name}.comet.tmp")
            examples = destination.with_name(f".{destination.name}.comet-inputs.json")
            log_path = ROOT_DIR / entry["log"]
            try:
                ensure_local_model_available(model)
                command, env = model_command(model, temporary, args.n_samples)
                env["COMET_DEFER_PATH"] = str(examples)
                entry.update(
                    status="running",
                    attempts=entry["attempts"] + 1,
                    started_at=now(),
                    ended_at=None,
                    last_error=None,
                )
                state["updated_at"] = now()
                atomic_json(STATE_PATH, state)
                active_names.add(name)
                model_started = time.time()
                next_progress = model_started + PROGRESS_INTERVAL_SECONDS
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(f"\n[{now()}] attempt {entry['attempts']} started\n")
                    log.flush()
                    child = subprocess.Popen(
                        command,
                        cwd=SCRIPT_DIR,
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    children.append(child)
                    while child.poll() is None:
                        time.sleep(10)
                        if time.time() >= next_progress:
                            print(progress(state, started), flush=True)
                            next_progress += PROGRESS_INTERVAL_SECONDS
                if child.returncode:
                    raise RuntimeError(f"evaluation exited with code {child.returncode}")
                if not temporary.exists() or not examples.exists():
                    raise RuntimeError("generation did not produce deferred COMET inputs")

                # The previous model scores on CPU while this model generates
                # on the llama.cpp GPU. Keep only one scorer active at a time.
                finish_score(pending_score)
                scorer_log = log_path.open("a", encoding="utf-8")
                scorer = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "score_comet.py",
                        "--examples",
                        str(examples),
                        "--result",
                        str(temporary),
                    ],
                    cwd=SCRIPT_DIR,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=scorer_log,
                    stderr=subprocess.STDOUT,
                )
                scorer_log.close()
                children.append(scorer)
                pending_score = (scorer, model, temporary, examples, model_started)
            except Exception as exc:
                entry.update(status="failed", ended_at=now(), last_error=str(exc))
                active_names.discard(name)
                temporary.unlink(missing_ok=True)
                examples.unlink(missing_ok=True)
            finally:
                state["updated_at"] = now()
                atomic_json(STATE_PATH, state)
                print(progress(state, started), flush=True)
        finish_score(pending_score)
        return 0 if all(item["status"] == "complete" for item in state["models"].values()) else 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
