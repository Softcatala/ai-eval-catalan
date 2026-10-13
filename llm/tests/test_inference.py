from inference import inference_params
import model


def test_gpt_6_uses_supported_reasoning_effort():
    params = inference_params(provider="openai", model_name="gpt-6-astra")

    assert params["reasoning_effort"] == "low"
    assert "temperature" not in params
    assert params["max_tokens"] >= 1024


def test_lm_eval_gpt_6_payload_omits_unsupported_parameters():
    adapter = object.__new__(model._lm_oai.OpenAIChatCompletion)
    adapter.model = "gpt-6-astra"
    adapter.base_url = "https://api.openai.com/v1/chat/completions"
    adapter._max_gen_toks = 256

    payload = adapter._create_payload(
        [{"role": "user", "content": "Hola"}],
        gen_kwargs={"max_tokens": 16, "reasoning_effort": "low"},
    )

    assert "stop" not in payload
    assert "temperature" not in payload
