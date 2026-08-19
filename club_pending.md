# CLUB QA F1 Pending Runs

Task: populate missing `benchmarks.club_qa.token_f1` values only. Do not
recompute results where `token_f1` is already present.

Pending existing result files:

```text
muse-glimmer-30b
claude-opus-4-7
```

Notes:

- `muse-glimmer-30b` is a local GGUF model served by the existing
  `llama-server` at `http://127.0.0.1:9090/v1`. Do not start another server.
  Its first missing-F1 retry returned an empty generation. The client now
  renders `llm/templates/muse_glimmer_user.jinja` into a direct-to-user
  `/completions` request; a live probe succeeded and the full retry is running.
- `claude-opus-4-7` is configured as
  `global.anthropic.claude-opus-4-7` through Amazon Bedrock. The OpenAI-style
  Bedrock path does not support this model, so the client now uses Bedrock's
  native Converse endpoint. A live probe succeeded and the full retry is
  running.
- `claude-sonnet-4-6` is not part of this task. Its result file does not exist,
  and the requested Claude model is Opus 4.7.
- `llm/recalculate_club_f1.py` runs one model at a time and merges only
  `benchmarks.club_qa` into an existing result JSON.

Local resume command:

```bash
cd llm
env PYTHONPATH=. uv run python -u recalculate_club_f1.py \
  --models muse-glimmer-30b \
  --llama-server-url http://127.0.0.1:9090/v1 \
  --model-timeout-seconds 2700
```

After completion, audit the configured results and retain only existing files
whose `benchmarks.club_qa.token_f1` is absent from the pending list.
