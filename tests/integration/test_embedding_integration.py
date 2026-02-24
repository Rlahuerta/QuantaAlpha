"""
Tests for embedding model integration.

Unit tests: verify provider-prefix stripping logic in APIBackend.
Integration tests (requires_api_key): verify live calls to local Ollama and
confirm rdagent's LiteLLMAPIBackend resolves correctly via the ollama/ prefix.
"""

from __future__ import annotations

import os
import types
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_provider_prefix(model: str) -> str:
    """Mirrors the logic added to APIBackend._create_embedding_inner_function."""
    if "/" in model:
        return model.split("/", 1)[1]
    return model


# ---------------------------------------------------------------------------
# Unit tests — no network required
# ---------------------------------------------------------------------------

class TestProviderPrefixStripping(unittest.TestCase):
    """Unit-test the prefix-stripping logic in isolation."""

    def test_ollama_prefix_stripped(self):
        self.assertEqual(_strip_provider_prefix("ollama/mxbai-embed-large"), "mxbai-embed-large")

    def test_no_prefix_unchanged(self):
        self.assertEqual(_strip_provider_prefix("mxbai-embed-large"), "mxbai-embed-large")

    def test_openai_prefix_stripped(self):
        self.assertEqual(_strip_provider_prefix("openai/text-embedding-3-small"), "text-embedding-3-small")

    def test_multi_slash_strips_only_first(self):
        self.assertEqual(_strip_provider_prefix("ollama/org/model"), "org/model")

    def test_empty_string_unchanged(self):
        self.assertEqual(_strip_provider_prefix(""), "")


class TestAPIBackendPrefixStripping(unittest.TestCase):
    """Verify APIBackend passes stripped model name to the OpenAI-compat client."""

    def _make_backend_with_model(self, embedding_model: str):
        """Build a minimal APIBackend-like object with mocked embedding_client."""
        from quantaalpha.llm.client import APIBackend  # import inside test to avoid side-effects at collection

        backend = object.__new__(APIBackend)
        backend.use_azure = False
        backend.use_embedding_cache = False
        backend.dump_embedding_cache = False
        backend.embedding_model = embedding_model

        # Fake embedding response
        fake_embedding = MagicMock()
        fake_embedding.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]

        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = fake_embedding
        backend.embedding_client = mock_client

        # Patch LLM_SETTINGS attributes accessed in the method
        import quantaalpha.llm.client as client_mod
        patcher_max = patch.object(client_mod.LLM_SETTINGS, "embedding_max_str_num", 50)
        patcher_wait = patch.object(client_mod.LLM_SETTINGS, "embedding_batch_wait_seconds", 0)
        patcher_max.start()
        patcher_wait.start()
        self.addCleanup(patcher_max.stop)
        self.addCleanup(patcher_wait.stop)

        return backend, mock_client

    def test_ollama_prefix_stripped_in_api_call(self):
        backend, mock_client = self._make_backend_with_model("ollama/mxbai-embed-large")
        backend._create_embedding_inner_function(["hello world"])
        called_model = mock_client.embeddings.create.call_args[1]["model"]
        self.assertEqual(called_model, "mxbai-embed-large",
                         "ollama/ prefix must be stripped before calling OpenAI-compat API")

    def test_no_prefix_model_unchanged_in_api_call(self):
        backend, mock_client = self._make_backend_with_model("mxbai-embed-large")
        backend._create_embedding_inner_function(["hello world"])
        called_model = mock_client.embeddings.create.call_args[1]["model"]
        self.assertEqual(called_model, "mxbai-embed-large")

    def test_azure_uses_full_model_name(self):
        """Azure paths must never strip the model name."""
        backend, mock_client = self._make_backend_with_model("ollama/mxbai-embed-large")
        backend.use_azure = True
        backend._create_embedding_inner_function(["hello world"])
        called_model = mock_client.embeddings.create.call_args[1]["model"]
        self.assertEqual(called_model, "ollama/mxbai-embed-large",
                         "Azure path must use the full model name, including any prefix")

    def test_returns_list_of_embeddings(self):
        backend, mock_client = self._make_backend_with_model("ollama/mxbai-embed-large")

        # Make the mock return one embedding item per input string
        def _fake_create(model, input, **kwargs):  # noqa: A002
            resp = MagicMock()
            resp.data = [MagicMock(embedding=[0.1, 0.2, 0.3]) for _ in input]
            return resp

        mock_client.embeddings.create.side_effect = _fake_create
        result = backend._create_embedding_inner_function(["a", "b"])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], [0.1, 0.2, 0.3])


class TestEnvEmbeddingModel(unittest.TestCase):
    """Verify .env sets EMBEDDING_MODEL with the ollama/ prefix."""

    def test_env_embedding_model_has_ollama_prefix(self):
        env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
        with open(env_path) as f:
            content = f.read()
        # Find the EMBEDDING_MODEL line
        for line in content.splitlines():
            if line.startswith("EMBEDDING_MODEL="):
                value = line.split("=", 1)[1].strip()
                self.assertTrue(
                    value.startswith("ollama/"),
                    f"EMBEDDING_MODEL should start with 'ollama/' for litellm routing, got: {value!r}",
                )
                return
        self.fail("EMBEDDING_MODEL not found in .env")


# ---------------------------------------------------------------------------
# Integration tests — require local Ollama running with mxbai-embed-large
# ---------------------------------------------------------------------------

@unittest.skipUnless(
    os.environ.get("EMBEDDING_BASE_URL") or True,  # always attempt; skip on timeout
    "Integration tests require local Ollama"
)
class TestEmbeddingIntegration(unittest.TestCase):
    """Live integration tests against local Ollama embedding API."""

    BASE_URL = "http://localhost:11434/v1"
    MODEL_BARE = "mxbai-embed-large"
    MODEL_PREFIXED = "ollama/mxbai-embed-large"

    def _ollama_available(self) -> bool:
        import urllib.request, urllib.error
        try:
            urllib.request.urlopen("http://localhost:11434", timeout=2)  # noqa: S310
            return True
        except Exception:
            return False

    def setUp(self):
        if not self._ollama_available():
            self.skipTest("Local Ollama not reachable at localhost:11434")

    def test_openai_client_bare_model_name(self):
        """QuantaAlpha path: OpenAI client + bare model name (no prefix)."""
        import openai
        client = openai.OpenAI(api_key="ollama", base_url=self.BASE_URL)
        resp = client.embeddings.create(model=self.MODEL_BARE, input=["quantaalpha factor test"])
        self.assertEqual(len(resp.data), 1)
        emb = resp.data[0].embedding
        self.assertIsInstance(emb, list)
        self.assertGreater(len(emb), 0, "Embedding vector must be non-empty")

    def test_openai_client_prefixed_model_name_fails(self):
        """Confirm that Ollama OpenAI-compat API rejects the ollama/ prefix (motivates the strip)."""
        import openai
        client = openai.OpenAI(api_key="ollama", base_url=self.BASE_URL)
        with self.assertRaises(Exception):
            client.embeddings.create(model=self.MODEL_PREFIXED, input=["test"])

    def test_litellm_prefixed_model_routes_to_ollama(self):
        """rdagent path: litellm with ollama/ prefix routes to localhost:11434."""
        from litellm import embedding
        resp = embedding(model=self.MODEL_PREFIXED, input=["quantaalpha factor test"])
        self.assertEqual(len(resp.data), 1)
        emb = resp.data[0]["embedding"]
        self.assertIsInstance(emb, list)
        self.assertGreater(len(emb), 0)

    def test_embedding_dimension_consistency(self):
        """Both paths must return the same embedding dimension."""
        import openai
        from litellm import embedding as litellm_embedding

        client = openai.OpenAI(api_key="ollama", base_url=self.BASE_URL)
        openai_resp = client.embeddings.create(model=self.MODEL_BARE, input=["test"])
        dim_openai = len(openai_resp.data[0].embedding)

        litellm_resp = litellm_embedding(model=self.MODEL_PREFIXED, input=["test"])
        dim_litellm = len(litellm_resp.data[0]["embedding"])

        self.assertEqual(dim_openai, dim_litellm,
                         f"Dimension mismatch: openai={dim_openai}, litellm={dim_litellm}")

    def test_api_backend_create_embedding_end_to_end(self):
        """Full QuantaAlpha APIBackend.create_embedding() with ollama/ prefix in EMBEDDING_MODEL."""
        os.environ.setdefault("OPENAI_API_KEY", "ollama")
        os.environ["EMBEDDING_MODEL"] = self.MODEL_PREFIXED
        os.environ["EMBEDDING_BASE_URL"] = self.BASE_URL
        os.environ["EMBEDDING_API_KEY"] = "ollama"

        # Re-create a fresh backend that picks up the env vars
        from quantaalpha.llm.client import APIBackend
        backend = APIBackend(embedding_model=self.MODEL_PREFIXED)
        result = backend.create_embedding(input_content="factor momentum test")
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    @unittest.skipUnless(
        os.environ.get("RUN_RDAGENT_EMBED_TEST"),
        "Set RUN_RDAGENT_EMBED_TEST=1 to run rdagent backend embedding test"
    )
    def test_rdagent_litellm_backend_create_embedding(self):
        """rdagent LiteLLMAPIBackend.create_embedding() with ollama/ prefix resolves correctly."""
        os.environ["EMBEDDING_MODEL"] = self.MODEL_PREFIXED

        from rdagent.oai.backend.litellm import LiteLLMAPIBackend
        backend = LiteLLMAPIBackend()
        result = backend.create_embedding(input_content=["momentum factor"])
        self.assertIsInstance(result, list)
        self.assertGreater(len(result[0]), 0)


if __name__ == "__main__":
    unittest.main()
