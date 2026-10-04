from benchmark import _match_server_model


def test_gemma_benchmark_does_not_match_another_provider_or_quantization():
    model = {
        "model": "gemma4-e4b-q8",
        "model_spec": "unsloth/gemma-4-E4B-it-GGUF:Q8_0",
    }
    wrong_ids = [
        "bartowski/google_gemma-4-E4B-it-GGUF:Q8_0",
        "bartowski/gemma-4-E4B-it-GGUF:Q8_0",
        "google_gemma-4-E4B-it-Q8_0",
        "gemma-4-E4B-it-Q4_K_M",
        "gemma-4-E4B-it",
        "Q8_0",
    ]
    for server_id in wrong_ids:
        assert _match_server_model(model, [server_id]) is None

    filename_id = "gemma-4-E4B-it-Q8_0"
    assert _match_server_model(model, [filename_id]) == filename_id
    assert (
        _match_server_model(model, wrong_ids + [filename_id, model["model_spec"]])
        == model["model_spec"]
    )
