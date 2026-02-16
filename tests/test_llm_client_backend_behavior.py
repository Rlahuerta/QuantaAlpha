from types import SimpleNamespace

import pytest

import quantaalpha.llm.client as llm_client_module
from quantaalpha.core.utils import SingletonBaseClass
from quantaalpha.llm.client import APIBackend


class FakeBadRequestError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _non_stream_response(content: str, finish_reason: str = "stop"):
    choice = SimpleNamespace(
        message=SimpleNamespace(content=content),
        finish_reason=finish_reason,
    )
    usage = SimpleNamespace(total_tokens=10, prompt_tokens=6, completion_tokens=4)
    return SimpleNamespace(choices=[choice], usage=usage)


def _stream_chunk(content: str, finish_reason: str | None = None):
    choice = SimpleNamespace(
        delta=SimpleNamespace(content=content),
        finish_reason=finish_reason,
    )
    return SimpleNamespace(choices=[choice])


@pytest.fixture(autouse=True)
def _reset_singletons():
    SingletonBaseClass._instance_dict.clear()
    yield
    SingletonBaseClass._instance_dict.clear()


@pytest.fixture
def _patch_llm_settings(monkeypatch, tmp_path):
    settings = llm_client_module.LLM_SETTINGS
    monkeypatch.setattr(settings, "use_gcr_endpoint", False)
    monkeypatch.setattr(settings, "use_azure", False)
    monkeypatch.setattr(settings, "log_llm_chat_content", False)
    monkeypatch.setattr(settings, "chat_stream", False)
    monkeypatch.setattr(settings, "chat_model", "test-chat")
    monkeypatch.setattr(settings, "reasoning_model", "test-reasoning")
    monkeypatch.setattr(settings, "chat_model_map", "{}")
    monkeypatch.setattr(settings, "openai_base_url", "http://localhost:11434/v1")
    monkeypatch.setattr(settings, "embedding_base_url", "http://localhost:11434/v1")
    monkeypatch.setattr(settings, "ollama_api_key", "dummy")
    monkeypatch.setattr(settings, "openai_api_key", "dummy")
    monkeypatch.setattr(settings, "embedding_api_key", "dummy")
    monkeypatch.setattr(settings, "prompt_cache_path", str(tmp_path / "prompt_cache.db"))
    monkeypatch.setattr(settings, "use_auto_chat_cache_seed_gen", False)
    monkeypatch.setattr(settings, "max_retry", 2)
    monkeypatch.setattr(settings, "retry_wait_seconds", 0)


@pytest.fixture
def _patch_openai(monkeypatch):
    class _DefaultOpenAI:
        def __init__(self, api_key=None, base_url=None):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: _non_stream_response('{"ok": true}')
                )
            )
            self.embeddings = SimpleNamespace(create=lambda **kwargs: SimpleNamespace(data=[]))

    fake_openai_module = SimpleNamespace(
        OpenAI=_DefaultOpenAI,
        BadRequestError=FakeBadRequestError,
    )
    monkeypatch.setattr(llm_client_module, "openai", fake_openai_module)


def _build_backend():
    return APIBackend(chat_api_key="dummy", embedding_api_key="dummy")


def _build_backend_without_explicit_keys():
    return APIBackend(chat_api_key=None, embedding_api_key=None)


def test_chat_completion_json_mode_normalizes_response(_patch_llm_settings, _patch_openai):
    backend = _build_backend()
    call_kwargs = {}

    def _fake_create(**kwargs):
        call_kwargs.update(kwargs)
        return _non_stream_response('{"score": 1,}')

    backend.chat_client.chat.completions.create = _fake_create
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "return json"},
    ]

    response, finish_reason = backend._create_chat_completion_inner_function(
        messages=messages,
        json_mode=True,
    )

    assert response == '{"score": 1}'
    assert finish_reason == "stop"
    assert call_kwargs["response_format"] == {"type": "json_object"}


def test_chat_completion_json_mode_keeps_invalid_text(_patch_llm_settings, _patch_openai):
    backend = _build_backend()
    backend.chat_client.chat.completions.create = lambda **kwargs: _non_stream_response("not-json")
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "return json"},
    ]

    response, _ = backend._create_chat_completion_inner_function(messages=messages, json_mode=True)

    assert response == "not-json"


def test_chat_completion_stream_mode_aggregates_chunks(monkeypatch, _patch_llm_settings, _patch_openai):
    monkeypatch.setattr(llm_client_module.LLM_SETTINGS, "chat_stream", True)
    backend = _build_backend()
    backend.chat_client.chat.completions.create = lambda **kwargs: [
        _stream_chunk("hel"),
        _stream_chunk("lo", finish_reason="stop"),
    ]
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "hello"},
    ]

    response, finish_reason = backend._create_chat_completion_inner_function(messages=messages)

    assert response == "hello"
    assert finish_reason == "stop"


def test_chat_completion_uses_cache_on_repeated_prompt(_patch_llm_settings, _patch_openai):
    backend = APIBackend(
        chat_api_key="dummy",
        embedding_api_key="dummy",
        use_chat_cache=True,
        dump_chat_cache=True,
    )
    create_calls = {"count": 0}

    def _fake_create(**kwargs):
        create_calls["count"] += 1
        return _non_stream_response("cached-response")

    backend.chat_client.chat.completions.create = _fake_create
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "same prompt"},
    ]

    first_response, first_finish_reason = backend._create_chat_completion_inner_function(
        messages=messages,
        seed=123,
    )
    second_response, second_finish_reason = backend._create_chat_completion_inner_function(
        messages=messages,
        seed=123,
    )

    assert first_response == "cached-response"
    assert first_finish_reason == "stop"
    assert second_response == "cached-response"
    assert second_finish_reason is None
    assert create_calls["count"] == 1


def test_retry_adds_json_prompt_hint_on_bad_request(_patch_llm_settings, _patch_openai):
    backend = _build_backend()
    call_kwargs = []

    def _fake_chat_auto_continue(**kwargs):
        call_kwargs.append(kwargs.copy())
        if len(call_kwargs) == 1:
            raise FakeBadRequestError("'messages' must contain the word 'json' in some form")
        return "ok"

    backend._create_chat_completion_auto_continue = _fake_chat_auto_continue

    result = backend._try_create_chat_completion_or_embedding(
        chat_completion=True,
        messages=[{"role": "user", "content": "hello"}],
    )

    assert result == "ok"
    assert "add_json_in_prompt" not in call_kwargs[0]
    assert call_kwargs[1]["add_json_in_prompt"] is True


def test_retry_truncates_embedding_inputs_on_context_error(_patch_llm_settings, _patch_openai):
    backend = _build_backend()
    seen_inputs = []

    def _fake_embedding_inner_function(**kwargs):
        current_input = kwargs["input_content_list"]
        seen_inputs.append(current_input[:])
        if len(seen_inputs) == 1:
            raise FakeBadRequestError("maximum context length")
        return ["ok"] * len(current_input)

    backend._create_embedding_inner_function = _fake_embedding_inner_function

    result = backend._try_create_chat_completion_or_embedding(
        embedding=True,
        input_content_list=["abcdef", "1234"],
    )

    assert seen_inputs[0] == ["abcdef", "1234"]
    assert seen_inputs[1] == ["abc", "12"]
    assert result == ["ok", "ok"]


def test_backend_prefers_ollama_api_key_from_env(monkeypatch, _patch_llm_settings, _patch_openai):
    monkeypatch.setattr(llm_client_module.LLM_SETTINGS, "ollama_api_key", "")
    monkeypatch.setattr(llm_client_module.LLM_SETTINGS, "openai_api_key", "")
    monkeypatch.setenv("OLLAMA_API_KEY", "ollama-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    backend = _build_backend_without_explicit_keys()

    assert backend.chat_api_key == "ollama-key"


def test_backend_falls_back_to_openai_api_key_env(monkeypatch, _patch_llm_settings, _patch_openai):
    monkeypatch.setattr(llm_client_module.LLM_SETTINGS, "ollama_api_key", "")
    monkeypatch.setattr(llm_client_module.LLM_SETTINGS, "openai_api_key", "")
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    backend = _build_backend_without_explicit_keys()

    assert backend.chat_api_key == "openai-key"
