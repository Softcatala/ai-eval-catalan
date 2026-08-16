"""Shared model repository URL helpers for eval JSON exports."""


def repo_url(model: str) -> str:
    model = model.removeprefix("global.anthropic.")
    repo = model.split(":", 1)[0]
    if "/" in repo:
        return f"https://huggingface.co/{repo}"
    if model.startswith("whisper-"):
        return f"https://huggingface.co/openai/{model}"
    if model.startswith("omniASR_"):
        return f"https://huggingface.co/facebook/{model.replace('_', '-')}"
    if model.startswith("gemma-4-"):
        return f"https://huggingface.co/google/{model}-it"
    if model == "gemini-embedding-001":
        return "https://ai.google.dev/gemini-api/docs/embeddings"
    if model.startswith("gemini-"):
        return "https://ai.google.dev/gemini-api/docs/models"
    if model.startswith(("gpt-", "text-embedding-")):
        return f"https://platform.openai.com/docs/models/{model}"
    if model.startswith("claude-"):
        return "https://docs.anthropic.com/en/docs/about-claude/models/overview"
    raise ValueError(f"No repo URL configured for model: {model}")
