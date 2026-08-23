import argparse
import json
import platform
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_evals import MODELS


DEFAULT_SERVER_URL = "http://127.0.0.1:9090/v1"
DEFAULT_OUTPUT = "benchmark.json"
BACKEND = "eval-http"
MAX_TOKENS = 256

BENCHMARK_PROMPT = """\
Respon en catala. Escriu una explicacio estructurada i detallada sobre com
avaluaries la qualitat d'un model de llenguatge en tasques de pregunta-resposta,
traduccio, resum i seguiment d'instruccions. Inclou criteris, exemples,
limitacions i recomanacions practiques. Mantingues un text continu i prou llarg
per mesurar la velocitat de generacio de manera estable.
"""


class BenchmarkError(Exception):
    pass


def _arg_value(args: list[str], name: str) -> str | None:
    try:
        return args[args.index(name) + 1]
    except (ValueError, IndexError):
        return None


def _is_gguf_model(model_spec: str) -> bool:
    lower = model_spec.lower()
    return "gguf" in lower or lower.endswith(".gguf")


def _local_models() -> list[dict[str, Any]]:
    models = []
    for entry in MODELS:
        model_spec = _arg_value(entry.get("args", []), "--model")
        if (
            entry.get("cloud")
            or entry.get("quantized_analysis_only")
            or not model_spec
            or not _is_gguf_model(model_spec)
        ):
            continue
        models.append(
            {
                "model": entry["display_name"],
                "model_spec": model_spec,
                "device": _arg_value(entry.get("args", []), "--device") or "cuda",
                "quantization": entry.get("quantization", ""),
            }
        )
    return models


def _expected_gguf_filename(model_spec: str) -> str | None:
    if model_spec.endswith(".gguf"):
        return Path(model_spec).name

    if ":" in model_spec:
        repo, quant = model_spec.rsplit(":", 1)
    else:
        repo, quant = model_spec, "Q8_0"

    filename_overrides = {
        "RichardErkhov/BSC-LT_-_salamandra-7b-instruct-gguf": "salamandra-7b-instruct.{quant}.gguf",
    }
    if repo in filename_overrides:
        return filename_overrides[repo].format(quant=quant)

    model_base = repo.split("/")[-1].replace("-GGUF", "")
    return f"{model_base}-{quant}.gguf"


def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _quant_key(value: str) -> str | None:
    match = re.search(r"(?<![a-z0-9])(q\d+(?:[_-]?[a-z0-9]+)*)", value.lower())
    return _compact(match.group(1)) if match else None


def _without_quant(value: str) -> str:
    lower = value.lower()
    lower = re.sub(r"(?<![a-z0-9])q\d+(?:[_-]?[a-z0-9]+)*", "", lower)
    lower = lower.replace(".gguf", "")
    lower = lower.replace("-gguf", "")
    lower = lower.replace("_gguf", "")
    lower = lower.replace("gguf", "")
    return lower.strip(" ._-:/")


def _name_variants(model: dict[str, Any]) -> set[str]:
    values = {model["model"], model["model_spec"]}
    filename = _expected_gguf_filename(model["model_spec"])
    if filename:
        values.add(filename)
        values.add(Path(filename).stem)

    repo = model["model_spec"].rsplit(":", 1)[0]
    repo_name = Path(repo).name
    values.add(repo_name)
    if "_" in repo_name:
        values.add(repo_name.split("_")[-1])

    variants = set()
    for value in values:
        cleaned = _without_quant(value)
        if cleaned:
            variants.add(cleaned)
        if "/" in cleaned:
            variants.add(cleaned.rsplit("/", 1)[-1])
    return {_compact(variant) for variant in variants if _compact(variant)}


def _match_server_model(model: dict[str, Any], server_model_ids: list[str]) -> str | None:
    model_quant = _quant_key(model["model_spec"])
    model_names = _name_variants(model)
    for server_id in server_model_ids:
        server_quant = _quant_key(server_id)
        if model_quant and server_quant and model_quant != server_quant:
            continue

        server_name = _compact(_without_quant(server_id.rsplit("/", 1)[-1]))
        if any(name in server_name or server_name in name for name in model_names):
            return server_id
    return None


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("runs", []), list):
        raise BenchmarkError(f"{path} does not contain a valid benchmark result object")
    data.setdefault("runs", [])
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def _run_optional(command: list[str], timeout: float = 3.0) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def _gpu_name() -> str | None:
    output = _run_optional(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return None
    gpus = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2:
            gpus.append(f"{parts[0]} {parts[1]} MiB")
        elif parts:
            gpus.append(parts[0])
    return "; ".join(gpus) if gpus else None


def _hardware_string(device: str) -> str:
    parts = []
    gpu = _gpu_name()
    if gpu:
        parts.append(gpu)
    parts.append(_cpu_name())
    parts.append(device)
    return ", ".join(parts)


def _http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 600.0,
) -> dict[str, Any]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise BenchmarkError(f"{method} {url} failed with HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise BenchmarkError(f"{method} {url} failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise BenchmarkError(f"{method} {url} timed out") from exc


def _server_model_ids(server_url: str) -> list[str]:
    data = _http_json("GET", f"{server_url.rstrip('/')}/models", timeout=5.0)
    ids = [
        str(model["id"])
        for model in data.get("data", [])
        if isinstance(model, dict) and model.get("id")
    ]
    if not ids:
        raise BenchmarkError("server did not return any model ids from /models")
    return ids


def _chat_completion(
    server_url: str,
    server_model: str,
    timeout: float,
) -> tuple[dict[str, Any], float]:
    payload = {
        "model": server_model,
        "messages": [{"role": "user", "content": BENCHMARK_PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
    }
    start = time.perf_counter()
    data = _http_json(
        "POST",
        f"{server_url.rstrip('/')}/chat/completions",
        payload=payload,
        timeout=timeout,
    )
    return data, time.perf_counter() - start


def _extract_metrics(data: dict[str, Any], elapsed_seconds: float) -> dict[str, Any]:
    timings = data.get("timings") if isinstance(data.get("timings"), dict) else {}
    predicted_n = timings.get("predicted_n")
    predicted_ms = timings.get("predicted_ms")
    prompt_n = timings.get("prompt_n")

    if predicted_n is not None and predicted_ms:
        generated_tokens = int(predicted_n)
        generation_seconds = float(predicted_ms) / 1000.0
        prompt_tokens = int(prompt_n) if prompt_n is not None else None
    else:
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        completion_tokens = usage.get("completion_tokens")
        if completion_tokens is None:
            raise BenchmarkError("response did not include generated token count")
        generated_tokens = int(completion_tokens)
        generation_seconds = elapsed_seconds
        prompt_tokens = usage.get("prompt_tokens")

    if generated_tokens <= 0:
        raise BenchmarkError("response generated zero tokens")
    if generation_seconds <= 0:
        raise BenchmarkError("generation time was zero")

    return {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "generation_seconds": round(generation_seconds, 4),
        "generation_tokens_per_sec": round(generated_tokens / generation_seconds, 2),
    }


def _benchmark_model(
    model: dict[str, Any],
    server_url: str,
    server_model: str,
    hardware: str,
    timeout: float,
) -> dict[str, Any]:
    _chat_completion(server_url, server_model=server_model, timeout=timeout)
    data, elapsed_seconds = _chat_completion(
        server_url, server_model=server_model, timeout=timeout
    )
    metrics = _extract_metrics(data, elapsed_seconds)
    return {
        "model": model["model"],
        "model_spec": model["model_spec"],
        "server_model": server_model,
        "backend": BACKEND,
        "device": model["device"],
        **metrics,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _completed_specs(data: dict[str, Any]) -> set[str]:
    return {
        str(run.get("model_spec"))
        for run in data.get("runs", [])
        if isinstance(run, dict) and run.get("model_spec")
    }


def _print_table(data: dict[str, Any]) -> None:
    runs = [run for run in data.get("runs", []) if isinstance(run, dict)]
    order = {model["model_spec"]: index for index, model in enumerate(_local_models())}
    runs.sort(key=lambda run: order.get(str(run.get("model_spec")), len(order)))

    rows = []
    for run in runs:
        speed = run.get("generation_tokens_per_sec")
        speed_text = f"{float(speed):.2f}" if isinstance(speed, (int, float)) else "?"
        rows.append((str(run.get("model", "?")), str(run.get("backend", "?")), speed_text))

    model_width = max([len("Model"), *(len(row[0]) for row in rows)])
    backend_width = max([len("Backend"), *(len(row[1]) for row in rows)])
    speed_width = max([len("Gen tok/s"), *(len(row[2]) for row in rows)])
    print(
        f"{'Model':<{model_width}}  "
        f"{'Backend':<{backend_width}}  "
        f"{'Gen tok/s':>{speed_width}}"
    )
    print(f"{'-' * model_width}  {'-' * backend_width}  {'-' * speed_width}")
    for model, backend, speed in rows:
        print(f"{model:<{model_width}}  {backend:<{backend_width}}  {speed:>{speed_width}}")

    device = next((run.get("device") for run in runs if run.get("device")), None)
    if device:
        print(f"\nDevice: `{device}`")
    hardware = data.get("hardware")
    if hardware:
        print(f"\nHardware: `{hardware}`")


def _print_skipped(skipped: list[tuple[str, str]]) -> None:
    if not skipped:
        return
    print("\nSkipped/unavailable models:")
    for name, reason in skipped:
        print(f"- {name}: {reason}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark local GGUF model generation speed")
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda", help="Device label used in hardware metadata")
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Timeout in seconds for each completion request",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Only print the table from the JSON output file",
    )
    args = parser.parse_args()

    output_path = Path(args.output)

    if args.print_only:
        data = _read_json(output_path)
        if data is None:
            print(f"No benchmark results found at {output_path}", file=sys.stderr)
            return 1
        _print_table(data)
        return 0

    local_models = _local_models()
    hardware = _hardware_string(args.device)
    data = _read_json(output_path) or {"hardware": hardware, "runs": []}

    recorded_hardware = data.get("hardware")
    if recorded_hardware != hardware:
        print("Hardware mismatch; stopping without modifying results.", file=sys.stderr)
        print(f"Recorded: {recorded_hardware}", file=sys.stderr)
        print(f"Current : {hardware}", file=sys.stderr)
        return 2

    completed = _completed_specs(data)
    missing = [model for model in local_models if model["model_spec"] not in completed]
    if not missing:
        _print_table(data)
        return 0

    skipped: list[tuple[str, str]] = []
    try:
        server_model_ids = _server_model_ids(args.server_url)
    except BenchmarkError as exc:
        server_model_ids = []
        for model in missing:
            skipped.append((model["model"], str(exc)))

    for model in missing:
        server_model = _match_server_model(model, server_model_ids) if server_model_ids else None
        if server_model_ids and not server_model:
            skipped.append((model["model"], "server is not serving this model"))
            continue
        if not server_model_ids:
            continue

        print(f"[RUN] {model['model']} ({server_model})")
        try:
            run = _benchmark_model(
                model,
                args.server_url,
                server_model=server_model,
                hardware=hardware,
                timeout=args.timeout,
            )
        except BenchmarkError as exc:
            skipped.append((model["model"], str(exc)))
            continue

        data["runs"].append(run)
        _write_json(output_path, data)
        print(
            f"[DONE] {model['model']} "
            f"({run['generation_tokens_per_sec']:.2f} tok/s) -> {output_path}"
        )

    _print_table(data)
    _print_skipped(skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
