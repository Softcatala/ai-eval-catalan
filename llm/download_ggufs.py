"""Download configured local GGUF models and write a llama-server preset."""

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

from models_config import MODELS
from model_specs import arg_value, expected_gguf_filename, is_gguf_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Download configured local GGUF models")
    parser.add_argument("--output-dir", required=True, help="Destination directory")
    parser.add_argument("--models", nargs="+", help="Optional display-name subset")
    parser.add_argument("--presets-file", help="Optional llama-server models-preset INI")
    parser.add_argument("--include-quantized-analysis", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = set(args.models or [])
    unknown = selected - {model["display_name"] for model in MODELS}
    if unknown:
        parser.error(f"unknown model display name(s): {', '.join(sorted(unknown))}")

    count = 0
    preset_lines = ["version = 1", "", "[*]", "ctx-size = 2048", ""]

    for model in MODELS:
        if selected and model["display_name"] not in selected:
            continue
        if model.get("quantized_analysis_only") and not args.include_quantized_analysis:
            continue
        model_spec = arg_value(model.get("args", []), "--model")
        if model.get("cloud") or not model_spec or not is_gguf_model(model_spec):
            continue

        if model_spec.lower().endswith(".gguf"):
            path = Path(model_spec).expanduser()
            if not path.exists():
                parser.error(
                    f"{model['display_name']}: local GGUF file not found: {model_spec}"
                )
        else:
            repo = model_spec.rsplit(":", 1)[0] if ":" in model_spec else model_spec
            filename = expected_gguf_filename(model_spec)
            print(f"[GET] {model['display_name']}: {repo}/{filename}")
            path = Path(
                hf_hub_download(repo_id=repo, filename=filename, local_dir=output_dir)
            )

        print(f"[OK] {model['display_name']}: {path}")
        preset_lines += [f"[{model_spec}]", f"model = {path.resolve().as_posix()}", ""]
        count += 1

    if count == 0:
        print("No local GGUF models selected.")
        return 0

    if args.presets_file:
        presets_path = Path(args.presets_file).expanduser()
        presets_path.parent.mkdir(parents=True, exist_ok=True)
        presets_path.write_text("\n".join(preset_lines), encoding="utf-8")
        print(f"[DONE] presets: {presets_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
