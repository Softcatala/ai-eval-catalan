# COMET evaluation task

## Goal

Rerun only FLORES translation (`n=400`) for every model in `llm/run_evals.py`,
replacing published BLEU scores with fresh COMET scores while preserving every
other benchmark result.

## Run

- Include local, cloud, and quantization-analysis models.
- Evaluate EN→CA, CA→EN, ES→CA, and CA→ES with
  `Unbabel/wmt22-comet-da` as configured in `llm/model.py`.
- Keep COMET on its native scale; never derive it from BLEU or divide by 100.
- Existing BLEU results are not complete and must be replaced.
- Merge the fresh FLORES block into each configured result JSON. Preserve its
  metadata and all existing non-translation metrics unchanged.
- Reuse `http://127.0.0.1:9090/v1` for local models; never start or stop
  llama.cpp. Run sequentially unless memory and server isolation make
  concurrency safe.
- Run cloud API models concurrently with local llama.cpp generation when both
  are pending. Limit local work to one generator at a time; cloud work must not
  block the local GPU queue or CPU COMET scoring.
- Use configured cloud credentials without logging or persisting secrets.
- Log each model separately and print progress at least every 10 minutes with:
  completed/total count, number and names of models left, running and failed
  models, elapsed time, and estimated ETA based on completed model runtimes.
- In every progress report, include a completed-model table with separate BLEU
  and COMET scores for EN→CA, CA→EN, ES→CA, and CA→ES, followed by a brief
  conclusion on ranking impact and any evaluation issues. Do not compare the
  raw BLEU and COMET scales as if they were equivalent.
- Continue independent models after failures, then retry failures once.

## Crash-safe resume

Implement a runner such as `run_comet_evals.py` with `--resume` and
`--force MODEL`.

- Store each model's status (`pending`, `running`, `complete`, or `failed`),
  attempts, timestamps, output, log, and last error in `comet_eval_state.json`.
- Atomically checkpoint every state transition: write and `fsync` a sibling
  temporary file, then rename it over the state file.
- Write results to temporary files and rename them into place only after JSON
  parsing and validation succeed.
- On startup/resume, validate completed outputs, skip valid COMET results,
  reset stale `running` entries, remove/quarantine partial files, and queue
  missing, invalid, or BLEU-only results.
- On `SIGINT`/`SIGTERM`, return active work to `pending` and flush state/logs.
- Rerun a completed model only with `--force MODEL`.

## Validate each model

- The existing result JSON parses; non-translation benchmarks remain unchanged.
- All four FLORES directions have `n=400` and finite numeric `comet,none`.
- FLORES contains no `bleu,none` or `bleu_stderr,none`.
- `flores_en_ca` and `flores_es_ca` equal their directional COMET means.
- CLAM uses one averaged translation contribution without `/100` scaling.

## Finish

- Regenerate `llm/llms.json`, `llm/llms_quantized.json`, HTML summaries, and
  `report.html`.
- Ensure active results and generated files contain no BLEU fields or labels;
  historical logs may remain unchanged.
- Run LLM tests, lint, JSON validation, and `git diff --check`.
- Report completed and failed models with failure-log paths and retry status.

## Acceptance

- Every configured model has a valid fresh `n=400` result using COMET.
- Results, summaries, and CLAM translation scores agree and publish no BLEU.
- Interrupting a running model and invoking `--resume` twice leaves state and
  completed outputs valid, skips completed models, and finishes remaining work.
