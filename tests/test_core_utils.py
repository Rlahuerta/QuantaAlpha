import pickle
from pathlib import Path

import pytest

import quantaalpha.core.utils as utils_module
from quantaalpha.core.utils import (
    CacheSeedGen,
    RDAgentException,
    SingletonBaseClass,
    _subprocess_wrapper,
    cache_with_pickle,
    import_class,
    multiprocessing_wrapper,
    parse_json,
    similarity,
)


class _DemoSingleton(SingletonBaseClass):
    pass


def test_singleton_requires_kwargs_and_is_unpicklable():
    with pytest.raises(RDAgentException):
        _DemoSingleton(1)

    first = _DemoSingleton(name="alpha")
    second = _DemoSingleton(name="alpha")
    assert first is second

    with pytest.raises(pickle.PicklingError):
        first.__reduce__()


def test_parse_similarity_and_import_helpers():
    assert parse_json('{"k": 1}') == {"k": 1}

    with pytest.raises(ValueError):
        parse_json("not-json")

    assert similarity("abc", "abc") == 100
    assert isinstance(similarity(None, "abc"), int)
    assert import_class("pathlib.Path") is Path


def test_seed_and_multiprocessing_wrappers(monkeypatch):
    seed_gen = CacheSeedGen()
    seed_gen.set_seed(123)
    assert isinstance(seed_gen.get_next_seed(), int)

    assert _subprocess_wrapper(lambda x, y: x + y, 7, [1, 2]) == 3

    direct = multiprocessing_wrapper([(lambda x: x + 1, (1,)), (lambda x: x + 2, (1,))], n=1)
    assert direct == [2, 3]

    class _FakeAsyncResult:
        def __init__(self, value):
            self._value = value

        def get(self):
            return self._value

    class _FakePool:
        def __init__(self, processes):
            self.processes = processes

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def apply_async(self, func, args=()):
            return _FakeAsyncResult(func(*args))

    monkeypatch.setattr(utils_module.mp, "Pool", _FakePool)
    pooled = multiprocessing_wrapper([(lambda x: x * 2, (2,)), (lambda x: x * 3, (2,))], n=2)
    assert pooled == [4, 6]


def test_cache_with_pickle_handles_direct_none_and_cached_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(utils_module.RD_AGENT_SETTINGS, "pickle_cache_folder_path_str", str(tmp_path))

    direct_calls = {"count": 0}

    @cache_with_pickle(lambda x: f"direct-{x}")
    def _direct(x):
        direct_calls["count"] += 1
        return x + 1

    monkeypatch.setattr(utils_module.RD_AGENT_SETTINGS, "cache_with_pickle", False)
    assert _direct(1) == 2
    assert _direct(1) == 2
    assert direct_calls["count"] == 2

    none_hash_calls = {"count": 0}

    @cache_with_pickle(lambda *args, **kwargs: None)
    def _none_hash(x):
        none_hash_calls["count"] += 1
        return x * 2

    monkeypatch.setattr(utils_module.RD_AGENT_SETTINGS, "cache_with_pickle", True)
    assert _none_hash(2) == 4
    assert _none_hash(2) == 4
    assert none_hash_calls["count"] == 2

    lock_entered = {"count": 0}

    class _FakeLock:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            lock_entered["count"] += 1

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(utils_module, "FileLock", _FakeLock)
    monkeypatch.setattr(utils_module.RD_AGENT_SETTINGS, "use_file_lock", True)
    cached_calls = {"count": 0}

    @cache_with_pickle(lambda x: f"cached-{x}", post_process_func=lambda x, cached_res: cached_res + x)
    def _cached(x):
        cached_calls["count"] += 1
        return x

    assert _cached(3) == 3
    assert _cached(3) == 6
    assert cached_calls["count"] == 1
    assert lock_entered["count"] == 1

    monkeypatch.setattr(utils_module.RD_AGENT_SETTINGS, "use_file_lock", False)
    no_lock_calls = {"count": 0}

    @cache_with_pickle(lambda x: f"nolock-{x}")
    def _no_lock(x):
        no_lock_calls["count"] += 1
        return x * 10

    assert _no_lock(4) == 40
    assert _no_lock(4) == 40
    assert no_lock_calls["count"] == 1
