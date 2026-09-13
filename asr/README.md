# ASR evaluation

Run all standard local and cloud evaluations (two local GPU jobs plus one cloud
job) with:

```bash
source /home/jupyter/.bashrc
CONDA_ENV= uv run python run_evals.py --device cuda --jobs 2 --overwrite
```

omniASR uses a separate uv environment because it requires a different Torch
version. Run each omniASR model with:

```bash
env -u CONDA_PREFIX CONDA_ENV= CUDA_VISIBLE_DEVICES=0 \
  uv run --project omni -- python hf-eval.py omniASR_CTC_300M \
  --device cuda --output evals/omni_ctc_300m.json
```

Replace the model name and output path for the other `omniASR_*` models.
`run_evals.py` does not yet route omniASR models to `omni/.venv` automatically.

Microsoft VibeVoice uses the main environment's optional `vibe` group:

```bash
CONDA_ENV= uv run --group vibe python hf-eval.py microsoft/VibeVoice-ASR-HF \
  --device cuda --output evals/vibevoice.json
```

VibeVoice requires Transformers 5.3 or newer. The optional group is pinned to
that range and the model-load smoke test passes with Transformers 5.17.0.
