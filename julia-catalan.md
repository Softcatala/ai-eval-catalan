# Julia-1: Catalan prompt sensitivity on MASSIVE

We observed low Catalan scenario-classification accuracy using Julia-1’s original Hugging Face weights and published Python model/encoding code, running locally in PyTorch FP32.

Your [published Catalan result](https://huggingface.co/SupersonicLabs/Julia-1/blob/main/metrics/accuracy-20260924.json) is **75.96%** on 2,974 examples. Our custom English prompt achieves **52.08%** on that full split with Q8 GGUF; we have not reproduced your exact protocol.

## Controlled evidence

Same 100 Catalan test examples, unchanged utterances and option order:

| Question / descriptions | Native FP32 | Q8 GGUF |
|---|---:|---:|
| Catalan / Catalan | 21% | 23% |
| Catalan / English | 58% | 56% |
| English / Catalan | 17% | 20% |
| English / English | 60% | 62% |

Native predictions changed from incorrect to correct on **41 examples**, and regressed on **two**, when both prompt components became English. With the Catalan prompt, native inference predicted `news` **74 times**, although only **one** example belonged to that category.

Strict encoding accepts the prompts; budgets of 256 and 512 produce identical predictions. BF16 GGUF also scores only 20% with the Catalan prompt. These checks point toward sensitivity to our descriptions rather than truncation or quantization. Language and wording effects remain intertwined.

## Reproduce

Download these [supporting files](jev/experiments/): `reproduce_julia.py`, `prompts.json`, and `sample-ids.json`. Keep them together, then run:

```bash
uv run --python 3.13 reproduce_julia.py
```

The script pins dependencies and checkpoint revision, verifies checkpoint/dataset SHA-256, and uses CPU FP32, eight threads, batch size eight, and 1,024-token context. Expected results: **21/100** and **60/100**. Its output records environment details and predictions. [Original paired evidence](jev/experiments/hypotheses-results.json) is attached.

Could you share your exact MASSIVE question, option descriptions/order, state serialization, and evaluation script/settings? We want to distinguish prompt sensitivity from a protocol mismatch.
