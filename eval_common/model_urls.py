"""Shared model repository URL helpers for eval JSON exports."""

import re


GEMMA4_MODEL_REPOS = (
    "google/gemma-4-E2B-it",
    "google/gemma-4-E4B-it",
    "google/gemma-4-12b-it",
    "google/gemma-4-26B-A4B-it",
)


def _gemma4_aliases(repo: str) -> set[str]:
    name = repo.rsplit("/", 1)[-1]
    family = name.removeprefix("gemma-4-").removesuffix("-it")
    size = family.split("-A", 1)[0]
    return {
        repo.lower(),
        name.lower(),
        f"gemma-4-{family}".lower(),
        f"gemma-4-{size}".lower(),
        f"gemma4-{family}".lower(),
        f"gemma4-{size}".lower(),
    }


GEMMA4_REPO_BY_ALIAS = {
    alias: repo for repo in GEMMA4_MODEL_REPOS for alias in _gemma4_aliases(repo)
}


def _normalized_model_key(model: str) -> str:
    key = model.removeprefix("global.anthropic.").split(":", 1)[0]
    if "/" in key:
        key = key.rsplit("/", 1)[-1]
    key = key.removeprefix("google_")
    key = key.removesuffix("-GGUF")
    key = re.sub(r"-(?:I?Q\d(?:_[A-Z0-9]+)+|q[248])$", "", key, flags=re.IGNORECASE)
    return key.lower()


def repo_url(model: str) -> str:
    model = model.removeprefix("global.anthropic.")
    repo = model.split(":", 1)[0]
    if gemma4_repo := GEMMA4_REPO_BY_ALIAS.get(_normalized_model_key(model)):
        return f"https://huggingface.co/{gemma4_repo}"
    if "/" in repo:
        return f"https://huggingface.co/{repo}"
    if model.startswith("whisper-"):
        return f"https://huggingface.co/openai/{model}"
    if model.startswith("omniASR_"):
        return f"https://huggingface.co/facebook/{model.replace('_', '-')}"
    if model == "gemini-embedding-001":
        return "https://ai.google.dev/gemini-api/docs/embeddings"
    if model.startswith("gemini-"):
        return "https://ai.google.dev/gemini-api/docs/models"
    if model.startswith(("gpt-", "text-embedding-")):
        return f"https://platform.openai.com/docs/models/{model}"
    if model.startswith("claude-"):
        return "https://docs.anthropic.com/en/docs/about-claude/models/overview"
    raise ValueError(f"No repo URL configured for model: {model}")
