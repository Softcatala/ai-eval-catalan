from inference import inference_params
import model
from unittest.mock import Mock

from bedrock import runtime_lm_model


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


def test_bedrock_runtime_uses_native_payload_and_text_only(monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-token")
    adapter = runtime_lm_model(
        "global.anthropic.claude-opus-5-5",
        "https://bedrock-runtime.us-east-2.amazonaws.com",
    )
    payload = adapter._create_payload(
        [{"role": "user", "content": "Hola"}],
        gen_kwargs={"max_tokens": 2048, "reasoning_effort": "none"},
    )

    assert adapter.base_url.endswith("/model/global.anthropic.claude-opus-5-5/invoke")
    assert adapter.header == {"Authorization": "Bearer test-token"}
    assert payload["anthropic_version"] == "bedrock-2023-05-31"
    assert not {"model", "temperature", "reasoning_effort"} & payload.keys()
    assert adapter.parse_generations(
        {
            "content": [
                {"type": "thinking", "thinking": "reasoning"},
                {"type": "text", "text": "Hola"},
            ]
        }
    ) == ["Hola"]


def test_native_response_is_accepted_without_retry():
    response = {"content": [{"type": "text", "text": "Hola"}]}
    operation = Mock(return_value=response)

    assert model._resilient_model_call(operation)(Mock(max_retries=3), []) == response
    operation.assert_called_once()
