# CLUB QA F1 Pending Runs

Task: finish recalculating `benchmarks.club_qa.token_f1` for the remaining LLM result JSON files.

Already handled on `clubqa_f1`:

- Cleaned untracked `results*.json` files from the repo worktree.
- Recalculated and merged CLUB QA F1 for the models that completed on this machine.
- `llm/recalculate_club_f1.py` exists as a helper runner that runs one model at a time, merges only `benchmarks.club_qa`, and continues after failures.

Pending configured models:

```text
gemma3-27b
qwen3-14b
phi-4
qwen3.5-9b
qwen3.8-27b
muse-glimmer-30b
llama3.1-8b
aya-expanse-8b
eurollm-9b
salamandra-7b
gemma4-26b
claude-sonnet-4-6
```

Notes:

- `claude-sonnet-4-6` is missing its result file: `llm/evals/results_claude_sonnet_4_6.json`.
- The local models above failed here after `llama-server` began returning HTTP 500 and then connection refused.
- Before rerunning, start a healthy `llama-server` exposing the model aliases from `llm/run_evals.py`, normally at `http://localhost:9090/v1`.

Resume command:

```bash
cd llm
env PYTHONPATH=. uv run python recalculate_club_f1.py \
  --models gemma3-27b qwen3-14b phi-4 qwen3.5-9b qwen3.8-27b \
  muse-glimmer-30b llama3.1-8b aya-expanse-8b eurollm-9b salamandra-7b gemma4-26b \
  --model-timeout-seconds 2700
```

After completion:

```bash
git status --short
git diff --stat -- llm/evals/results*.json
```
