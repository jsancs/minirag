from minirag.utils.model_utils import handle_model
from minirag.utils.backend_manager import set_backend, get_backend_instance
from minirag.backends import get_backend


class TestModelUtils:
    def test_handle_model_ollama_missing(self, mocker):
        set_backend("ollama")
        backend = get_backend_instance()
        mocker.patch.object(backend, "check_model_exists", return_value=False)
        mock_pull = mocker.patch.object(backend, "pull_model")

        handle_model("missing_model")
        mock_pull.assert_called_once_with("missing_model")

    def test_handle_model_ollama_exists(self, mocker):
        set_backend("ollama")
        backend = get_backend_instance()
        mocker.patch.object(backend, "check_model_exists", return_value=True)
        mock_pull = mocker.patch.object(backend, "pull_model")

        handle_model("existing_model")
        mock_pull.assert_not_called()

    def test_handle_model_openai_missing(self, mocker, capsys, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        set_backend("openai")
        backend = get_backend_instance()
        mocker.patch.object(backend, "check_model_exists", return_value=False)
        mock_pull = mocker.patch.object(backend, "pull_model")

        handle_model("gpt-5-nano")
        mock_pull.assert_not_called()

        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "gpt-5-nano" in captured.out
