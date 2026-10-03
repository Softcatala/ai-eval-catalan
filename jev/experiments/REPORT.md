# Julia-1 accuracy investigation

All prompt comparisons use the same 100 MASSIVE 1.1 test IDs, seed 42.
Input is Catalan unless explicitly marked English. Native weights SHA256:
`df853bf7fe424420011f3d0c47a05d7341aa9eefa7fb9f203ea4aada4ad95b72`,
matching the publisher's benchmark. Native inference uses the published model.py
and data.py, CPU FP32, maximum length 1024, batches of 8. BF16 GGUF uses a
separate CPU llama.cpp server; Q8 uses the existing GPU server. Consequently
precision comparisons also change execution backend; matching native results
provide an independent check. Latencies here should not be compared.

| Hypothesis | Experiment | Result | Assessment |
|---|---|---|---|
| Catalan descriptions hurt | Cross question and description languages | Q8: CA/CA 23%, CA/EN 56%, EN/CA 20%, EN/EN 62% | Strong effect of descriptions; smaller effect of question language |
| Long descriptions are truncated | Native strict encoding; head budget 256 vs 512 | Strict head checks pass; every variant has identical predictions across budgets | Not the cause for these prompts |
| Shorter labels fix performance | Short descriptions / bare scenario names | Q8: short CA 28%, short EN 37%, bare EN 44% | Small CA improvement; brevity alone does not fix it |
| Option order affects results | Reverse all options with wording unchanged | Q8: CA 23% to 19%; EN 62% to 50% | Order sensitivity exists |
| Q8 precision causes collapse | BF16 GGUF, native FP32 using same prompts | CA: 23%, 20%, 21%; EN: 62%, 59%, 60% respectively | Does not explain the large gap |
| GGUF implementation/conversion causes collapse | Native publisher implementation and verified weights | Same low CA accuracy and similar EN accuracy | Does not explain collapse with our prompt |
| Our sample is unusually difficult | Full 2,974 Catalan examples, English question/descriptions | 52.08%, versus 62% on selected 100 | Original sample was easier; sampling does not explain low results |
| Catalan input itself is the dominant issue | Parallel English input on same 100 IDs, English question/descriptions | English 59%, Catalan 62% | No evidence in this comparison |
| Wrong checkpoint | Native checkpoint SHA256 compared with published result | Exact match | Ruled out for native comparison |
| Exact publisher MASSIVE prompt/protocol differs | Inspect released scripts and metrics | Exact MASSIVE prompt/reproduction script not found | Remains unresolved; no direct reproduction claim |

## Conclusion

The collapse is reproducible in the original native implementation. Our label
wording/language strongly affects Julia's decisions. Quantization, GGUF
conversion, truncation and sample difficulty do not explain the collapse.
With our full Catalan prompt, Q8 predicts `news` for 59 of 100 examples; native
FP32 predicts it for 74, confirming a strongly skewed output distribution.
Changing descriptions to English reduces this collapse, but our English prompt
still scores only 52.08% on the full Catalan test split, below the published
75.96%. The remaining 23.88-point gap cannot be attributed conclusively without
the publisher's exact MASSIVE request construction and evaluation protocol.

The earlier suggestion that token truncation might explain the gap is not
supported by these experiments. Do not present 23% as general Catalan ability,
or claim to have reproduced the published benchmark.

Raw paired predictions: `hypotheses-results.json`. Exact prompts: `prompts.json`.
Published scores: https://huggingface.co/SupersonicLabs/Julia-1/blob/main/metrics/accuracy-20260924.json
Native serialization: https://huggingface.co/SupersonicLabs/Julia-1/blob/main/julia/data.py
