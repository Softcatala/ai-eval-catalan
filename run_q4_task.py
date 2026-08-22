#!/usr/bin/env python3
"""Resumable runner for eval_q4.md."""

from __future__ import annotations

import html
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LLM = ROOT / "llm"
STATE = ROOT / "eval_q4_state.json"
REPORT = ROOT / "report.html"
SERVER = "http://localhost:9090/v1"

MODELS = {
    "qwen3-14b": ("results_qwen3_14b_q4.json", "results_qwen3_14b.json", "Qwen_Qwen3-14B-Q4_K_M"),
    "mistral-small-24b": ("results_mistral_small_24b_q4.json", "results_mistral_small_24b.json", "mistralai_Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M"),
    "phi-4": ("results_phi4_q4.json", "results_phi4_q8.json", "phi-4-Q4_K_M"),
    "qwen3.5-9b": ("results_qwen3.5_9b_q4.json", "results_qwen3.5_9b.json", "Qwen_Qwen3.5-9B-Q4_K_M"),
    "qwen3.8-27b": ("results_qwen3.8_27b.json", "results_qwen3.6_27b.json", "Qwen3.8-27B-Q4_K_M"),
    "muse-glimmer-30b": ("results_muse_glimmer_30b_q4.json", "results_muse_glimmer_30b_q8.json", "Muse-Glimmer-30B-UD-Q4_K_XL"),
    "llama3.1-8b": ("results_llama3.1_8b_q4.json", "results_llama3.1_8b.json", "Meta-Llama-3.1-8B-Instruct-Q4_K_M"),
    "aya-expanse-8b": ("results_aya_expanse_8b_q4.json", "results_aya_expanse_8b.json", "aya-expanse-8b-Q4_K_M"),
    "eurollm-9b": ("results_eurollm_9b_q4.json", "results_eurollm_9b.json", "EuroLLM-9B-Instruct-Q4_K_M"),
    "salamandra-7b": ("results_salamandra_7b_q4.json", "results_salamandra_7b.json", "salamandra-7b-instruct-2606.Q4_K_M"),
}
PARALLEL = ["qwen3.5-9b", "llama3.1-8b", "aya-expanse-8b", "eurollm-9b", "salamandra-7b"]
SEQUENTIAL = ["mistral-small-24b", "phi-4", "qwen3.8-27b", "muse-glimmer-30b"]
FLORES_TASKS = [
    "catalan_bench_flores_en-ca", "catalan_bench_flores_ca-en",
    "catalan_bench_flores_es-ca", "catalan_bench_flores_ca-es",
]
CLAM_METRICS = [
    "STS Pearson", "CatCoLA MCC", "CLUB QA EM", "CaSum ROUGE-L",
    "IFEval strict",
]


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_state() -> dict[str, str]:
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    return {name: state.get(name, "pending") for name in MODELS}


def metrics(data: dict) -> dict[str, float]:
    b = data.get("benchmarks", {})
    out = {}
    simple = {
        "STS Pearson": ("sts_ca", "pearson"),
        "CatCoLA MCC": ("catcola", "mcc"),
        "CLUB QA EM": ("club_qa", "exact_match_approx"),
        "CaSum ROUGE-L": ("casum", "rougeL"),
        "IFEval strict": ("ifeval", "prompt_level_strict_acc,none"),
    }
    for label, (bench, key) in simple.items():
        value = b.get(bench, {}).get(key)
        if isinstance(value, (int, float)):
            out[label] = float(value)
    flores = b.get("flores", {})
    for task in FLORES_TASKS:
        value = flores.get(task, {}).get("comet,none")
        if isinstance(value, (int, float)):
            out[f"FLORES {task.removeprefix('catalan_bench_flores_')} COMET"] = float(value)
    return out


def valid(path: Path) -> bool:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    b = data.get("benchmarks", {})
    if set(b) != {"sts_ca", "casum", "catcola", "club_qa", "flores", "ifeval"}:
        return False
    if any(isinstance(value, dict) and "error" in value for value in b.values()):
        return False
    values = metrics(data)
    return (
        len(values) == 9
        and all(value > 0 for value in values.values())
        and all(b[name].get("n") == 400 for name in ("sts_ca", "casum", "catcola", "club_qa"))
    )


def baseline_metrics(path: Path) -> dict[str, float]:
    if path.exists():
        return metrics(json.loads(path.read_text()))
    relative = str(path.relative_to(ROOT))
    found = subprocess.run(
        ["git", "show", f"HEAD:{relative}"], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    return metrics(json.loads(found.stdout)) if found.returncode == 0 else {}


def clam_score(values: dict[str, float]) -> float | None:
    scores = [values[name] for name in CLAM_METRICS if name in values]
    en_ca = [values.get("FLORES en-ca COMET"), values.get("FLORES ca-en COMET")]
    es_ca = [values.get("FLORES es-ca COMET"), values.get("FLORES ca-es COMET")]
    translation = []
    if all(value is not None for value in en_ca):
        translation.append(sum(en_ca) / len(en_ca))
    if all(value is not None for value in es_ca):
        translation.append(sum(es_ca) / len(es_ca))
    if translation:
        scores.append(sum(translation) / len(translation))
    return sum(scores) / len(scores) * 100 if scores else None


def render_report(state: dict[str, str]) -> None:
    sections = []
    for name in MODELS:
        q4_name, q8_name, _ = MODELS[name]
        q4 = LLM / "evals" / q4_name
        if state.get(name) != "completed" or not valid(q4):
            continue
        current = metrics(json.loads(q4.read_text()))
        baseline_path = LLM / "evals" / q8_name
        baseline = baseline_metrics(baseline_path)
        rows = []
        current_clam = clam_score(current)
        baseline_clam = clam_score(baseline)
        if current_clam is not None:
            old_text = "—" if baseline_clam is None else f"{baseline_clam:.4f}"
            delta = "—" if baseline_clam is None else f"{current_clam - baseline_clam:+.4f}"
            rows.append(f"<tr><td>CLAM</td><td>{current_clam:.4f}</td><td>{old_text}</td><td>{delta}</td></tr>")
        for metric, value in current.items():
            old = baseline.get(metric)
            delta = "—" if old is None else f"{value - old:+.4f}"
            old_text = "—" if old is None else f"{old:.4f}"
            rows.append(f"<tr><td>{html.escape(metric)}</td><td>{value:.4f}</td><td>{old_text}</td><td>{delta}</td></tr>")
        sections.append(f"<h2>{html.escape(name)}</h2><table><thead><tr><th>Metric</th><th>Q4</th><th>Q8 baseline</th><th>Delta</th></tr></thead><tbody>{''.join(rows)}</tbody></table>")
    document = "<!doctype html><html><head><meta charset='utf-8'><title>Q4 evaluation report</title><style>body{font-family:sans-serif;margin:2rem}table{border-collapse:collapse}th,td{border:1px solid #ccc;padding:.4rem .7rem;text-align:right}th:first-child,td:first-child{text-align:left}</style></head><body><h1>Q4 evaluation report</h1>" + "".join(sections) + "</body></html>\n"
    tmp = REPORT.with_suffix(".html.tmp")
    tmp.write_text(document)
    os.replace(tmp, REPORT)


def finish(name: str, state: dict[str, str]) -> None:
    q4_name, q8_name, _ = MODELS[name]
    q4 = LLM / "evals" / q4_name
    if not valid(q4):
        state[name] = "pending"
        atomic_json(STATE, state)
        raise RuntimeError(f"{name}: output failed validation")
    state[name] = "completed"
    atomic_json(STATE, state)
    render_report(state)
    subprocess.run(["git", "add", str(q4.relative_to(ROOT))], cwd=ROOT, check=True)
    q8 = LLM / "evals" / q8_name
    if q8 != q4 and q8.exists():
        subprocess.run(["git", "rm", str(q8.relative_to(ROOT))], cwd=ROOT, check=True)


def progress(state: dict[str, str]) -> None:
    done = [n for n, status in state.items() if status == "completed"]
    left = [n for n, status in state.items() if status != "completed"]
    print(f"[PROGRESS] completed {len(done)}/{len(state)}: {', '.join(done) or 'none'}", flush=True)
    print(f"[PROGRESS] left {len(left)}/{len(state)}: {', '.join(left) or 'none'}", flush=True)


def command(name: str) -> list[str]:
    _, _, server_model = MODELS[name]
    return [
        str(LLM / ".venv" / "bin" / "python"), "-u", "run_evals.py",
        "--models", name, "--n-samples", "400", "--benchmarks", "all",
        "--server-url", SERVER, "--server-model", server_model,
    ]


def warm_model(name: str) -> None:
    """Wait for the router to load a model before scored requests begin."""
    model = MODELS[name][2]
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Respon: sí"}],
        "temperature": 0,
        "max_tokens": 2,
    }).encode()
    deadline = time.monotonic() + 1200
    while True:
        request = urllib.request.Request(
            f"{SERVER}/chat/completions", data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                json.loads(response.read())
            print(f"[READY] {name}", flush=True)
            return
        except Exception as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"{name}: server did not become ready: {exc}")
            print(f"[WAIT] {name}: model is loading", flush=True)
            time.sleep(10)


def run_group(names: list[str], state: dict[str, str], parallel: bool) -> None:
    pending = [n for n in names if state[n] != "completed"]
    while pending:
        # The existing router is configured with models-max=2. Two-wide batches
        # retain parallel execution without producing slot-exhaustion errors.
        batch = pending[:2] if parallel else pending[:1]
        processes = {}
        for name in batch:
            state[name] = "running"
            atomic_json(STATE, state)
        for name in batch:
            warm_model(name)
        for name in batch:
            log_path = ROOT / f"run_q4_{name}.log"
            log_start = log_path.stat().st_size if log_path.exists() else 0
            log = log_path.open("a")
            print(f"[START] {name}", flush=True)
            processes[name] = (subprocess.Popen(command(name), cwd=LLM, stdout=log, stderr=subprocess.STDOUT), log, log_path, log_start)
        last_progress = time.monotonic()
        failures = []
        while processes:
            for name, (proc, log, log_path, log_start) in list(processes.items()):
                rc = proc.poll()
                if rc is None:
                    continue
                log.close()
                del processes[name]
                run_log = log_path.read_text(errors="replace")[log_start:]
                request_errors = "llama-server generation failed" in run_log or "llama-server request failed" in run_log
                if rc == 0 and not request_errors:
                    try:
                        finish(name, state)
                        print(f"[DONE] {name}", flush=True)
                    except RuntimeError as exc:
                        print(f"[RETRY] {exc}", flush=True)
                        failures.append(name)
                else:
                    output = LLM / "evals" / MODELS[name][0]
                    if output.exists():
                        output.unlink()
                    state[name] = "pending"
                    atomic_json(STATE, state)
                    reason = "server request errors" if request_errors else f"exit {rc}"
                    print(f"[RETRY] {name}: {reason}", flush=True)
                    failures.append(name)
            if time.monotonic() - last_progress >= 1800:
                progress(state)
                last_progress = time.monotonic()
            if processes:
                time.sleep(10)
        progress(state)
        pending = failures + [n for n in pending if n not in batch]
        if failures:
            time.sleep(15)


def main() -> int:
    state = load_state()
    for name, status in list(state.items()):
        path = LLM / "evals" / MODELS[name][0]
        if status == "completed" and not valid(path):
            state[name] = "pending"
        elif status == "running":
            state[name] = "pending"
    atomic_json(STATE, state)
    for name in list(state):
        if state[name] == "completed":
            finish(name, state)
    progress(state)
    run_group(PARALLEL, state, parallel=True)
    run_group(SEQUENTIAL, state, parallel=False)
    progress(state)
    return 0


def watch_report(parent_pid: int) -> int:
    """Re-render with the latest generator after the long-lived runner updates state."""
    last_mtime = -1
    while True:
        try:
            os.kill(parent_pid, 0)
        except ProcessLookupError:
            return 0
        mtime = STATE.stat().st_mtime_ns
        if mtime != last_mtime:
            time.sleep(5)
            render_report(load_state())
            last_mtime = mtime
        time.sleep(10)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--watch-report":
        sys.exit(watch_report(int(sys.argv[2])))
    sys.exit(main())
