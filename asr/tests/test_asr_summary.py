import json
import sys

from asr.summarize_results import main


def test_export_strips_aina_prefix_and_preserves_repository_url(tmp_path, monkeypatch):
    model = "projecte-aina/whisper-large-v3-ca-3catparla"
    (tmp_path / "model.json").write_text(json.dumps({"model": model}))
    output = tmp_path / "asrs.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_results",
            "--results-dir",
            str(tmp_path),
            "--json-out",
            str(output),
        ],
    )

    main()

    row = json.loads(output.read_text())["data"][0]
    assert row["model"] == "whisper-large-v3-ca-3catparla"
    assert row["repo_url"] == f"https://huggingface.co/{model}"
