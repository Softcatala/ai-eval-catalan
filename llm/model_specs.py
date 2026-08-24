from pathlib import Path


FILENAME_OVERRIDES = {
    "RichardErkhov/BSC-LT_-_salamandra-7b-instruct-gguf": "salamandra-7b-instruct.{quant}.gguf",
    "mradermacher/salamandra-7b-instruct-2606-GGUF": "salamandra-7b-instruct-2606.{quant}.gguf",
}


def arg_value(args: list[str], name: str) -> str | None:
    try:
        return args[args.index(name) + 1]
    except (ValueError, IndexError):
        return None


def is_gguf_model(model_spec: str) -> bool:
    lower = model_spec.lower()
    return "gguf" in lower or lower.endswith(".gguf")


def expected_gguf_filename(model_spec: str, default_quant: str = "Q4_K_M") -> str:
    """Return the GGUF filename expected for a repo:quant spec or local .gguf path."""
    if model_spec.lower().endswith(".gguf"):
        return Path(model_spec).name

    if ":" in model_spec:
        repo, quant = model_spec.rsplit(":", 1)
    else:
        repo, quant = model_spec, default_quant

    if repo in FILENAME_OVERRIDES:
        return FILENAME_OVERRIDES[repo].format(quant=quant)

    model_base = repo.split("/")[-1].replace("-GGUF", "")
    return f"{model_base}-{quant}.gguf"
