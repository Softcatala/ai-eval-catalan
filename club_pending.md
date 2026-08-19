# CLUB QA F1 Pending Runs

Task: populate missing `benchmarks.club_qa.token_f1` values only. Do not
recompute results where `token_f1` is already present.

Pending existing result files: none.

All configured result files that exist now contain
`benchmarks.club_qa.token_f1`.

Notes:

- `muse-glimmer-30b` completed after the client was changed to render
  `llm/templates/muse_glimmer_user.jinja` into a direct-to-user `/completions`
  request. Its CLUB QA token F1 is `0.6617` (`n=400`).
- `claude-opus-4-7` is configured as
  `global.anthropic.claude-opus-4-7` through Amazon Bedrock. The OpenAI-style
  Bedrock path does not support this model, so the client now uses Bedrock's
  native Converse endpoint. It completed with CLUB QA token F1 `0.8039`
  (`n=400`).
- `claude-sonnet-4-6` is not part of this task. Its result file does not exist,
  and the requested Claude model is Opus 4.7.
- `llm/recalculate_club_f1.py` runs one model at a time and merges only
  `benchmarks.club_qa` into an existing result JSON.

After completion, audit the configured results and retain only existing files
whose `benchmarks.club_qa.token_f1` is absent from the pending list.
