# omniASR environment

This isolated uv project keeps omniASR's Torch/fairseq2 requirements separate
from the main ASR evaluation environment.

Run an omniASR evaluation from the repository root:

```bash
env -u CONDA_PREFIX CONDA_ENV= uv run --project omni -- python hf-eval.py omniASR_CTC_300M --device cuda --output evals/omni_ctc_300m.json
```
