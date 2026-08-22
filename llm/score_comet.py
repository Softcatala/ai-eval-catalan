"""Score deferred FLORES generations with COMET on CPU."""

import argparse
import json
import os
from pathlib import Path

from comet_config import COMET_CHECKPOINT, drop_legacy_translation_metrics
from model import _load_comet_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()

    examples = json.loads(args.examples.read_text(encoding="utf-8"))
    result = json.loads(args.result.read_text(encoding="utf-8"))
    flores = result["benchmarks"]["flores"]
    result["flores_comet_checkpoint"] = COMET_CHECKPOINT
    comet_model = _load_comet_model()
    for task, task_examples in examples.items():
        prediction = comet_model.predict(task_examples, batch_size=8, gpus=0)
        flores[task].pop("comet_pending", None)
        drop_legacy_translation_metrics(flores[task])
        flores[task]["comet,none"] = float(prediction.system_score)
        print(f"{task}: COMET={flores[task]['comet,none']:.4f}", flush=True)

    temporary = args.result.with_name(f".{args.result.name}.scored.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, args.result)
    args.examples.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
