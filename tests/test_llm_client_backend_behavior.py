import json
from types import SimpleNamespace

import numpy as np
import pytest

import quantaalpha.llm.client as llm_client_module
from quantaalpha.core.utils import SingletonBaseClass
from quantaalpha.llm.client import (
    APIBackend,
    ChatSession,
    ConvManager,
    SQliteLazyCache,
    SessionChatHistoryCache,
    calculate_embedding_distance_between_str_list,
)


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


def test_conv_manager_rotates_files_and_appends_latest(tmp_path):
    conv_dir = tmp_path / "llm_conv"
    conv_dir.mkdir(parents=True, exist_ok=True)
    (conv_dir / "0.json").write_text(json.dumps([["u0"], "a0"]), encoding="utf-8")
    (conv_dir / "1.json").write_text(json.dumps([["u1"], "a1"]), encoding="utf-8")

    manager = ConvManager(path=conv_dir, recent_n=2)
    manager.append((["new"], "answer"))

    assert (conv_dir / "0.json").exists()
    assert (conv_dir / "1.json").exists()
    assert (conv_dir / "2.json").exists()
    assert json.loads((conv_dir / "0.json").read_text(encoding="utf-8")) == [["new"], "answer"]


def test_sqlite_lazy_cache_roundtrip(tmp_path):
    cache = SQliteLazyCache(cache_location=str(tmp_path / "cache.db"))

    assert cache.chat_get("missing") is None
    assert cache.embedding_get("missing") is None
    assert cache.message_get("missing") == []

    cache.chat_set("k", "v")
    cache.embedding_set({"emb-k": [0.1, 0.2]})
    cache.message_set("cid", [{"role": "user", "content": "hello"}])

    assert cache.chat_get("k") == "v"
    assert cache.embedding_get("emb-k") == [0.1, 0.2]
    assert cache.message_get("cid")[-1]["content"] == "hello"


def test_chat_session_builds_and_persists_history(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_client_module.LLM_SETTINGS, "prompt_cache_path", str(tmp_path / "session_cache.db"))
    monkeypatch.setattr(llm_client_module.LLM_SETTINGS, "default_system_prompt", "default-system")

    class _FakeBackend:
        def calculate_token_from_messages(self, messages):
            return len(messages)

        def _try_create_chat_completion_or_embedding(self, **kwargs):
            return "assistant-reply"

    session = ChatSession(_FakeBackend(), conversation_id="cid-1", system_prompt="sys-1")
    prompt_messages = session.build_chat_completion_message("hello")
    token_count = session.build_chat_completion_message_and_calculate_token("world")
    response = session.build_chat_completion("hello")
    history = SessionChatHistoryCache().message_get("cid-1")

    assert prompt_messages[0]["role"] == "system"
    assert token_count == 2
    assert response == "assistant-reply"
    assert history[-1]["role"] == "assistant"
    assert history[-1]["content"] == "assistant-reply"


def test_build_messages_shrink_breaks_and_limit_history(monkeypatch, _patch_llm_settings, _patch_openai):
    monkeypatch.setattr(llm_client_module.LLM_SETTINGS, "max_past_message_include", 1)
    backend = _build_backend()

    messages = backend.build_messages(
        user_prompt="line1\n\n\nline2",
        system_prompt="sys\n\n\nprompt",
        former_messages=[
            {"role": "assistant", "content": "old-1"},
            {"role": "assistant", "content": "old-2"},
        ],
        shrink_multiple_break=True,
    )

    assert messages[0]["content"] == "sys\n\nprompt"
    assert messages[1]["content"] == "old-2"
    assert messages[-1]["content"] == "line1\n\nline2"


def test_auto_continue_when_finish_reason_is_length(_patch_llm_settings, _patch_openai):
    backend = _build_backend()
    calls = []

    def _fake_inner(messages, **kwargs):
        calls.append(messages)
        if len(calls) == 1:
            return "part-1", "length"
        return "part-2", "stop"

    backend._create_chat_completion_inner_function = _fake_inner
    result = backend._create_chat_completion_auto_continue(messages=[{"role": "user", "content": "q"}])

    assert result == "part-1part-2"
    assert calls[1][-1]["content"] == "continue the former output with no overlap"


def test_build_log_messages_truncates_content(_patch_llm_settings, _patch_openai):
    backend = _build_backend()
    log_text = backend._build_log_messages(
        [{"role": "user", "content": "x" * 120}],
        max_prompt_length=10,
    )

    assert "Role:" in log_text
    assert "... [120 chars]" in log_text


def test_build_messages_and_create_chat_completion_forwards_kwargs(_patch_llm_settings, _patch_openai):
    backend = _build_backend()
    captured = {}

    def _fake_try_create(**kwargs):
        captured.update(kwargs)
        return "ok"

    backend._try_create_chat_completion_or_embedding = _fake_try_create
    result = backend.build_messages_and_create_chat_completion(
        user_prompt="hello",
        former_messages=[{"role": "assistant", "content": "before"}],
        chat_cache_prefix="cache-key",
        json_mode=True,
    )

    assert result == "ok"
    assert captured["chat_completion"] is True
    assert captured["chat_cache_prefix"] == "cache-key"
    assert captured["messages"][-1]["content"] == "hello"


def test_embedding_inner_function_batches_and_uses_cache(monkeypatch, _patch_llm_settings, _patch_openai):
    monkeypatch.setattr(llm_client_module.LLM_SETTINGS, "embedding_max_str_num", 2)
    monkeypatch.setattr(llm_client_module.LLM_SETTINGS, "embedding_batch_wait_seconds", 0)

    backend = APIBackend(
        chat_api_key="dummy",
        embedding_api_key="dummy",
        use_embedding_cache=True,
        dump_embedding_cache=True,
    )
    call_inputs = []

    def _fake_embedding_create(model, input):
        call_inputs.append(list(input))
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[float(len(text)), float(i)]) for i, text in enumerate(input)]
        )

    backend.embedding_client.embeddings.create = _fake_embedding_create
    items = ["a", "bb", "ccc"]
    first = backend._create_embedding_inner_function(input_content_list=items)
    second = backend._create_embedding_inner_function(input_content_list=items)

    assert len(first) == 3
    assert first == second
    assert call_inputs == [["a", "bb"], ["ccc"]]


def test_calculate_embedding_distance_between_lists(monkeypatch):
    class _FakeBackend:
        def create_embedding(self, _content):
            return [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

    monkeypatch.setattr(llm_client_module, "APIBackend", lambda: _FakeBackend())

    similarity = calculate_embedding_distance_between_str_list(["src"], ["t1", "t2"])

    assert similarity[0][0] == pytest.approx(0.0)
    assert similarity[0][1] == pytest.approx(1 / np.sqrt(2))
    assert calculate_embedding_distance_between_str_list([], ["x"]) == [[]]


def test_chat_session_get_id_and_display_history():
    class _FakeBackend:
        def _try_create_chat_completion_or_embedding(self, **kwargs):
            return "ok"

        def calculate_token_from_messages(self, messages):
            return len(messages)

    session = ChatSession(_FakeBackend(), conversation_id="cid-static", system_prompt="sys")
    assert session.get_conversation_id() == "cid-static"
    assert session.display_history() is None


def test_conv_manager_rotate_unlinks_existing_target(tmp_path):
    conv_dir = tmp_path / "llm_conv_rotate"
    conv_dir.mkdir(parents=True, exist_ok=True)
    (conv_dir / "0.json").write_text(json.dumps([["u0"], "a0"]), encoding="utf-8")
    (conv_dir / "1.json").write_text(json.dumps([["u1"], "a1"]), encoding="utf-8")
    (conv_dir / "2.json").write_text(json.dumps([["stale"], "old"]), encoding="utf-8")

    manager = ConvManager(path=conv_dir, recent_n=2)
    manager._rotate_files()

    assert json.loads((conv_dir / "2.json").read_text(encoding="utf-8")) == [["u1"], "a1"]
    assert json.loads((conv_dir / "1.json").read_text(encoding="utf-8")) == [["u0"], "a0"]


def test_gcr_backend_init_and_chat_completion_path(monkeypatch):
    settings = SimpleNamespace(
        use_gcr_endpoint=True,
        gcr_endpoint_type="phi2",
        phi2_endpoint_key="gcr-key",
        phi2_endpoint_deployment="deploy",
        phi2_endpoint="https://example.com/invoke",
        gcr_endpoint_temperature=0.2,
        gcr_endpoint_top_p=0.9,
        gcr_endpoint_do_sample=True,
        gcr_endpoint_max_token=64,
        chat_model_map="{}",
        chat_model="gcr-default",
        dump_chat_cache=False,
        use_chat_cache=False,
        dump_embedding_cache=False,
        use_embedding_cache=False,
        prompt_cache_path=":memory:",
        retry_wait_seconds=0,
        use_auto_chat_cache_seed_gen=False,
        log_llm_chat_content=True,
        chat_temperature=0.2,
        chat_max_tokens=32,
        chat_frequency_penalty=0.0,
        chat_presence_penalty=0.0,
        chat_stream=False,
        chat_seed=None,
        reasoning_model="gcr-reasoning",
    )
    monkeypatch.setattr(llm_client_module, "LLM_SETTINGS", settings)

    logs = []
    monkeypatch.setattr(llm_client_module.logger, "info", lambda *args, **kwargs: logs.append(args[0]))

    class _Resp:
        def read(self):
            return b'{"output":"gcr-response"}'

    monkeypatch.setattr(llm_client_module.urllib.request, "urlopen", lambda req: _Resp())

    backend = APIBackend(chat_model="gcr-model")
    response, finish_reason = backend._create_chat_completion_inner_function(
        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}],
        reasoning_flag=False,
    )

    assert backend.headers["Authorization"] == "Bearer gcr-key"
    assert response == "gcr-response"
    assert finish_reason is None
    assert any("Response:" in str(item) for item in logs)


def test_gcr_backend_invalid_type_raises(monkeypatch):
    settings = SimpleNamespace(use_gcr_endpoint=True, gcr_endpoint_type="not-supported")
    monkeypatch.setattr(llm_client_module, "LLM_SETTINGS", settings)

    with pytest.raises(ValueError, match="Invalid gcr_endpoint_type"):
        APIBackend(chat_model="x")


def test_azure_backend_uses_token_provider(monkeypatch, _patch_llm_settings, _patch_openai):
    settings = llm_client_module.LLM_SETTINGS
    monkeypatch.setattr(settings, "use_azure", True)
    monkeypatch.setattr(settings, "chat_use_azure_token_provider", True)
    monkeypatch.setattr(settings, "embedding_use_azure_token_provider", True)
    monkeypatch.setattr(settings, "managed_identity_client_id", "mi-client")
    monkeypatch.setattr(settings, "chat_azure_api_base", "https://chat.azure")
    monkeypatch.setattr(settings, "chat_azure_api_version", "2024-01-01")
    monkeypatch.setattr(settings, "embedding_azure_api_base", "https://emb.azure")
    monkeypatch.setattr(settings, "embedding_azure_api_version", "2024-01-01")

    class _AzureOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=lambda **k: _non_stream_response('{"ok": true}'))
            )
            self.embeddings = SimpleNamespace(create=lambda **k: SimpleNamespace(data=[]))

    monkeypatch.setattr(llm_client_module.openai, "AzureOpenAI", _AzureOpenAI, raising=False)
    monkeypatch.setattr(
        llm_client_module,
        "DefaultAzureCredential",
        lambda **kwargs: SimpleNamespace(kwargs=kwargs),
        raising=False,
    )
    monkeypatch.setattr(
        llm_client_module,
        "get_bearer_token_provider",
        lambda credential, scope: f"token:{scope}",
        raising=False,
    )

    backend = APIBackend(chat_api_key="chat-k", embedding_api_key="emb-k")
    assert backend.chat_client.kwargs["azure_ad_token_provider"].startswith("token:")
    assert backend.embedding_client.kwargs["azure_ad_token_provider"].startswith("token:")


def test_azure_backend_uses_api_keys_when_token_provider_disabled(monkeypatch, _patch_llm_settings, _patch_openai):
    settings = llm_client_module.LLM_SETTINGS
    monkeypatch.setattr(settings, "use_azure", True)
    monkeypatch.setattr(settings, "chat_use_azure_token_provider", False)
    monkeypatch.setattr(settings, "embedding_use_azure_token_provider", False)
    monkeypatch.setattr(settings, "ollama_api_key", "")
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(settings, "embedding_openai_api_key", "")
    monkeypatch.setattr(settings, "embedding_api_key", "")
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.setattr(settings, "chat_azure_api_base", "https://chat.azure")
    monkeypatch.setattr(settings, "chat_azure_api_version", "2024-01-01")
    monkeypatch.setattr(settings, "embedding_azure_api_base", "https://emb.azure")
    monkeypatch.setattr(settings, "embedding_azure_api_version", "2024-01-01")

    class _AzureOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=lambda **k: _non_stream_response('{"ok": true}'))
            )
            self.embeddings = SimpleNamespace(create=lambda **k: SimpleNamespace(data=[]))

    monkeypatch.setattr(llm_client_module.openai, "AzureOpenAI", _AzureOpenAI, raising=False)

    backend = APIBackend(chat_api_key="chat-k", embedding_api_key="emb-k")
    assert backend.chat_client.kwargs["api_key"] == "chat-k"
    assert backend.embedding_client.kwargs["api_key"] == "emb-k"


def test_get_encoder_fallback_and_error_paths(monkeypatch, _patch_llm_settings, _patch_openai):
    backend = _build_backend()
    backend.chat_model = "gpt_4o"

    def _encoding_for_model(model):
        if model == "gpt_4o":
            raise KeyError("missing")
        if model == "gpt-4o":
            return "enc"
        raise KeyError("missing")

    monkeypatch.setattr(llm_client_module.tiktoken, "encoding_for_model", _encoding_for_model)
    assert backend._get_encoder() == "enc"

    # When all tiktoken lookups fail, _get_encoder falls back to cl100k_base
    monkeypatch.setattr(
        llm_client_module.tiktoken,
        "encoding_for_model",
        lambda model: (_ for _ in ()).throw(KeyError("missing")),
    )
    fallback = backend._get_encoder()
    assert fallback.name == "cl100k_base"


def test_build_chat_session_create_embedding_and_token_helper_defaults(_patch_llm_settings, _patch_openai):
    backend = _build_backend()
    backend._try_create_chat_completion_or_embedding = lambda **kwargs: [[1.0], [2.0]]

    session = backend.build_chat_session(conversation_id="conv-1", session_system_prompt="sys")
    assert isinstance(session, ChatSession)
    assert session.conversation_id == "conv-1"

    assert backend.create_embedding("s1") == [1.0]
    assert backend.create_embedding(["s1", "s2"]) == [[1.0], [2.0]]
    assert backend.build_messages_and_calculate_token("u", "s") > 0


def test_auto_continue_returns_single_response_when_not_length(_patch_llm_settings, _patch_openai):
    backend = _build_backend()
    backend._create_chat_completion_inner_function = lambda messages, **kwargs: ("done", "stop")
    assert backend._create_chat_completion_auto_continue(messages=[{"role": "user", "content": "q"}]) == "done"


def test_try_create_raises_runtime_after_generic_exceptions(monkeypatch, _patch_llm_settings, _patch_openai):
    backend = _build_backend()
    monkeypatch.setattr(llm_client_module.LLM_SETTINGS, "max_retry", 2)
    monkeypatch.setattr(llm_client_module.time, "sleep", lambda _seconds: None)
    backend._create_chat_completion_auto_continue = lambda **kwargs: (_ for _ in ()).throw(Exception("boom"))

    with pytest.raises(RuntimeError, match="Failed to create chat completion"):
        backend._try_create_chat_completion_or_embedding(
            chat_completion=True,
            messages=[{"role": "user", "content": "x"}],
        )


def test_embedding_inner_function_qwen_batches_azure_and_wait(monkeypatch, _patch_llm_settings, _patch_openai):
    monkeypatch.setattr(llm_client_module.LLM_SETTINGS, "embedding_max_str_num", 10)
    monkeypatch.setattr(llm_client_module.LLM_SETTINGS, "embedding_batch_wait_seconds", 0.01)
    backend = _build_backend()
    backend.use_azure = True
    backend.use_embedding_cache = False
    backend.embedding_model = "qwen-text-embedding-v4"

    sleeps = []
    monkeypatch.setattr(llm_client_module.time, "sleep", lambda seconds: sleeps.append(seconds))

    def _fake_embedding_create(model, input):
        return SimpleNamespace(data=[SimpleNamespace(embedding=[float(len(text))]) for text in input])

    backend.embedding_client = SimpleNamespace(embeddings=SimpleNamespace(create=_fake_embedding_create))
    result = backend._create_embedding_inner_function(["a", "bb", "ccc", "dddd"])

    assert [item[0] for item in result] == [1.0, 2.0, 3.0, 4.0]
    assert sleeps == [0.01]


def test_chat_completion_cache_logging_and_auto_seed(monkeypatch, _patch_llm_settings, _patch_openai):
    monkeypatch.setattr(llm_client_module.LLM_SETTINGS, "log_llm_chat_content", True)
    monkeypatch.setattr(llm_client_module.LLM_SETTINGS, "use_auto_chat_cache_seed_gen", True)
    backend = APIBackend(
        chat_api_key="dummy",
        embedding_api_key="dummy",
        use_chat_cache=True,
        dump_chat_cache=True,
    )

    monkeypatch.setattr(llm_client_module.LLM_CACHE_SEED_GEN, "get_next_seed", lambda: 77)
    logs = []
    monkeypatch.setattr(llm_client_module.logger, "info", lambda *args, **kwargs: logs.append(args[0]))
    backend.chat_client.chat.completions.create = lambda **kwargs: _non_stream_response("cached-body")
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}]

    first, _ = backend._create_chat_completion_inner_function(messages=messages, seed=None)
    second, _ = backend._create_chat_completion_inner_function(messages=messages, seed=None)

    assert first == "cached-body"
    assert second == "cached-body"
    assert any("Response(cached):" in str(msg) for msg in logs)


def test_chat_completion_add_json_hint_uses_tag_fallback(monkeypatch, _patch_llm_settings, _patch_openai):
    monkeypatch.setattr(llm_client_module.LLM_SETTINGS, "chat_model_map", '{"external_caller":"mapped-model"}')
    backend = _build_backend()
    backend.chat_model_map = {"external_caller": "mapped-model"}

    captured = {}
    backend.chat_client.chat.completions.create = lambda **kwargs: captured.update(kwargs) or _non_stream_response('{"x":1}')
    fake_stack = [
        SimpleNamespace(frame=SimpleNamespace(f_locals={}), function="f0"),
        SimpleNamespace(frame=SimpleNamespace(f_locals={}), function="f1"),
        SimpleNamespace(frame=SimpleNamespace(f_locals={}), function="f2"),
        SimpleNamespace(frame=SimpleNamespace(f_locals={}), function="f3"),
        SimpleNamespace(frame=SimpleNamespace(f_locals={}), function="external_caller"),
    ]
    monkeypatch.setattr(llm_client_module.inspect, "stack", lambda: fake_stack)

    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}]
    response, _ = backend._create_chat_completion_inner_function(
        messages=messages,
        json_mode=True,
        add_json_in_prompt=True,
        reasoning_flag=False,
    )

    assert response == '{"x": 1}'
    assert captured["model"] == "mapped-model"
    assert "Please respond in json format." in messages[0]["content"]
    assert "Please respond in json format." in messages[1]["content"]


def test_chat_completion_stream_and_non_stream_logging(monkeypatch, _patch_llm_settings, _patch_openai):
    logs = []
    monkeypatch.setattr(llm_client_module.logger, "info", lambda *args, **kwargs: logs.append(args[0]))
    monkeypatch.setattr(llm_client_module.LLM_SETTINGS, "log_llm_chat_content", True)
    backend = _build_backend()

    monkeypatch.setattr(llm_client_module.LLM_SETTINGS, "chat_stream", True)
    backend.chat_stream = True
    backend.chat_client.chat.completions.create = lambda **kwargs: [
        _stream_chunk("he"),
        _stream_chunk("llo", finish_reason="stop"),
    ]
    stream_resp, stream_finish = backend._create_chat_completion_inner_function(
        messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        reasoning_flag=False,
    )
    assert stream_resp == "hello"
    assert stream_finish == "stop"

    monkeypatch.setattr(llm_client_module.LLM_SETTINGS, "chat_stream", False)
    backend.chat_stream = False
    backend.chat_client.chat.completions.create = lambda **kwargs: _non_stream_response("world")
    non_stream_resp, non_stream_finish = backend._create_chat_completion_inner_function(
        messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        reasoning_flag=False,
    )
    assert non_stream_resp == "world"
    assert non_stream_finish == "stop"
    assert any("Response:" in str(msg) for msg in logs)
    assert any('"total_tokens": 10' in str(msg) for msg in logs)
