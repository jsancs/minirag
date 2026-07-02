import pytest
from minirag.backends.openai_backend import (
    _openai_available as openai_available_for_tests,
)
from minirag.utils.backend_manager import set_backend, get_backend_instance, _backend
from minirag.backends import get_backend, BACKENDS
from minirag.backends.base import Backend


class TestBackendManager:
    def test_set_backend_and_get_backend_instance(self):
        set_backend("ollama")
        backend = get_backend_instance()
        assert isinstance(backend, Backend)
        assert backend.__class__.__name__ == "OllamaBackend"

    def test_get_backend_instance_default(self):
        global _backend
        _backend = None
        backend = get_backend_instance()
        assert isinstance(backend, Backend)


class TestBackendRegistry:
    def test_get_backend_ollama(self):
        backend = get_backend("ollama")
        assert isinstance(backend, Backend)
        assert backend.__class__.__name__ == "OllamaBackend"

    def test_get_backend_invalid(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("invalid_backend")


class TestOllamaBackend:
    def test_generate_embeddings(self, mocker):
        backend = get_backend("ollama")
        mock_embeddings = mocker.patch(
            "ollama.embeddings", return_value={"embedding": [0.1, 0.2]}
        )

        result = backend.generate_embeddings("test text")
        assert isinstance(result, list)
        assert len(result) == 2
        assert result == [0.1, 0.2]
        mock_embeddings.assert_called_once_with(
            model="all-minilm",
            prompt="test text",
        )

    def test_generate_embeddings_defaults_model(self, mocker):
        backend = get_backend("ollama")
        mock_embeddings = mocker.patch(
            "ollama.embeddings", return_value={"embedding": [0.1, 0.2]}
        )

        backend.generate_embeddings("test text", "")

        mock_embeddings.assert_called_once_with(model="all-minilm", prompt="test text")

    def test_check_model_exists(self, mocker):
        backend = get_backend("ollama")
        mocker.patch("ollama.list", return_value={"models": [{"name": "llama3.2"}]})

        result = backend.check_model_exists("llama3.2")
        assert result is True

    def test_check_model_not_exists(self, mocker):
        backend = get_backend("ollama")
        mocker.patch("ollama.list", return_value={"models": []})

        result = backend.check_model_exists("unknown")
        assert result is False

    def test_chat_streaming_uses_deterministic_temperature(self, mocker):
        backend = get_backend("ollama")
        mock_chat = mocker.patch(
            "ollama.chat",
            return_value=iter([{"message": {"content": "hello"}}]),
        )

        result = list(backend.chat_streaming("llama3.2:1b", [{"role": "user", "content": "hi"}]))

        assert result == ["hello"]
        mock_chat.assert_called_once_with(
            model="llama3.2:1b",
            messages=[{"role": "user", "content": "hi"}],
            options={"temperature": 0},
            stream=True,
        )


@pytest.mark.skipif(
    not openai_available_for_tests,
    reason="OpenAI backend is not available (openai package not installed)",
)
class TestOpenAIBackend:
    def test_init_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            from minirag.backends.openai_backend import OpenAIBackend

            OpenAIBackend()

    def test_init_with_api_key(self, mocker, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        mock_openai = mocker.patch("minirag.backends.openai_backend.OpenAI")
        from minirag.backends.openai_backend import OpenAIBackend

        backend = OpenAIBackend()
        mock_openai.assert_called_once_with(
            api_key="test_key", base_url="https://api.openai.com/v1"
        )
        assert backend is not None

    def test_init_with_custom_base_url(self, mocker, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://custom.api.com/v1")
        mock_openai = mocker.patch("minirag.backends.openai_backend.OpenAI")
        from minirag.backends.openai_backend import OpenAIBackend

        backend = OpenAIBackend()
        mock_openai.assert_called_once_with(
            api_key="test_key", base_url="https://custom.api.com/v1"
        )
        assert backend is not None

    def test_init_with_empty_base_url(self, mocker, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        monkeypatch.setenv("OPENAI_BASE_URL", "")
        mock_openai = mocker.patch("minirag.backends.openai_backend.OpenAI")
        from minirag.backends.openai_backend import OpenAIBackend

        backend = OpenAIBackend()
        mock_openai.assert_called_once_with(
            api_key="test_key", base_url="https://api.openai.com/v1"
        )
        assert backend is not None

    def test_generate_embeddings(self, mocker, monkeypatch):
        mock_client = mocker.MagicMock()
        mock_response = mocker.MagicMock()
        mock_response.data = [mocker.MagicMock(embedding=[0.1, 0.2])]
        mock_client.embeddings.create.return_value = mock_response

        mocker.patch("minirag.backends.openai_backend.OpenAI", return_value=mock_client)
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")

        from minirag.backends.openai_backend import OpenAIBackend

        backend = OpenAIBackend()
        result = backend.generate_embeddings("test text")
        assert result == [0.1, 0.2]
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input="test text",
        )

    def test_chat_streaming_uses_deterministic_temperature(self, mocker, monkeypatch):
        mock_client = mocker.MagicMock()
        mock_chunk = mocker.MagicMock()
        mock_chunk.choices[0].delta.content = "hello"
        mock_client.chat.completions.create.return_value = iter([mock_chunk])

        mocker.patch("minirag.backends.openai_backend.OpenAI", return_value=mock_client)
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")

        from minirag.backends.openai_backend import OpenAIBackend

        backend = OpenAIBackend()
        result = list(backend.chat_streaming("gpt-4.1-mini", [{"role": "user", "content": "hi"}]))

        assert result == ["hello"]
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0,
            stream=True,
        )

    def test_check_model_exists(self, mocker, monkeypatch):
        mock_client = mocker.MagicMock()
        mock_client.models.retrieve.return_value = mocker.MagicMock()

        mocker.patch("minirag.backends.openai_backend.OpenAI", return_value=mock_client)
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")

        from minirag.backends.openai_backend import OpenAIBackend

        backend = OpenAIBackend()
        result = backend.check_model_exists("gpt-4")
        assert result is True

    def test_check_model_not_exists(self, mocker, monkeypatch):
        mock_client = mocker.MagicMock()
        mock_client.models.retrieve.side_effect = Exception()

        mocker.patch("minirag.backends.openai_backend.OpenAI", return_value=mock_client)
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")

        from minirag.backends.openai_backend import OpenAIBackend

        backend = OpenAIBackend()
        result = backend.check_model_exists("unknown")
        assert result is False
