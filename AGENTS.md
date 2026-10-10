# Model-provider preference

For Gemma GGUF models, prefer repositories published by `unsloth` over
equivalent repositories published by other providers. Before changing a model
source, verify that the target Unsloth repository provides the required
quantization and use its exact filename/quantization identifier.

# Code changes

Make the requested changes with the minimum number of code changes possible.

# Model quantization

Within each benchmark category (LLM, JEV or ASR), use the same quantization
scheme for every local model so results are comparable. Use `Q4_K_M` for
LLM models and `Q8_0` for JEV decision models. Dynamic quantizations such as
`UD-Q4_K_M` and `UD-Q4_K_XL` are distinct schemes, not interchangeable with
`Q4_K_M`. Keep comparisons of different quantizations separate from the
category's main benchmark. For ASR, verify and use the category's common
quantization scheme before adding a model.
