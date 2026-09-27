from model_specs import arg_value, expected_gguf_filename, is_gguf_model


def test_arg_value_returns_option_value():
    assert (
        arg_value(["--model", "repo/model", "--device", "cuda"], "--model")
        == "repo/model"
    )
    assert arg_value(["--model"], "--model") is None
    assert arg_value([], "--model") is None


def test_is_gguf_model_detects_repo_and_file_specs():
    assert is_gguf_model("bartowski/model-GGUF:Q4_K_M")
    assert is_gguf_model("/models/model.gguf")
    assert not is_gguf_model("openai")


def test_expected_gguf_filename_uses_quant_and_overrides():
    assert expected_gguf_filename("org/Model-GGUF:Q5_K_M") == "Model-Q5_K_M.gguf"
    assert expected_gguf_filename("org/Model-GGUF") == "Model-Q4_K_M.gguf"
    assert (
        expected_gguf_filename("org/Model-GGUF", default_quant="Q8_0")
        == "Model-Q8_0.gguf"
    )
    assert (
        expected_gguf_filename("BSC-LT/salamandra-7b-fc-2607-GGUF:Q4_0")
        == "ALIA-7b-fc-2607-Q4_0.gguf"
    )
    assert (
        expected_gguf_filename("CohereLabs/tiny-aya-water-GGUF:Q4_K_M")
        == "tiny-aya-water-q4_k_m.gguf"
    )
    assert (
        expected_gguf_filename("unsloth/GLM-4.7-Flash-GGUF:Q4_K_M")
        == "GLM-4.7-Flash-Q4_K_M.gguf"
    )
    assert (
        expected_gguf_filename("unsloth/Qwen3.5-4B-GGUF:Q4_K_M")
        == "Qwen3.5-4B-Q4_K_M.gguf"
    )
    assert (
        expected_gguf_filename(
            "mradermacher/ALIA-40b-instruct-2605-GGUF:Q4_K_M"
        )
        == "ALIA-40b-instruct-2605.Q4_K_M.gguf"
    )
