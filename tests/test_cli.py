import minirag.cli as cli


class TestCli:
    def session(self):
        return cli.ChatSession()

    def test_show_help_lists_core_commands(self, capsys):
        cli.show_help()

        captured = capsys.readouterr()

        assert "MiniRAG commands:" in captured.out
        assert "--top-k" in captured.out
        assert "/add" in captured.out
        assert "/activate <collection_name>" in captured.out
        assert "/clear" in captured.out
        assert "/retrieve <query>" in captured.out
        assert "/status" in captured.out

    def test_normalize_doc_path_strips_matching_quotes(self):
        assert cli.normalize_doc_path("'/tmp/document.pdf'") == "/tmp/document.pdf"
        assert cli.normalize_doc_path('"/tmp/document.pdf"') == "/tmp/document.pdf"
        assert cli.normalize_doc_path("/tmp/document.pdf") == "/tmp/document.pdf"

    def test_clear_command_clears_conversation_and_terminal(self, mocker, capsys):
        clear_conversation = mocker.patch("minirag.cli.clear_conversation")
        clear_terminal = mocker.patch("minirag.cli.clear_terminal")
        session = self.session()

        cli.handle_user_query("/clear", "llama3.1:8b", session)

        clear_conversation.assert_called_once_with(session)
        clear_terminal.assert_called_once_with()
        captured = capsys.readouterr()
        assert "Conversation cleared." in captured.out

    def test_activate_without_collection_name_prints_usage(self, capsys):
        cli.handle_user_query("/activate", "llama3.1:8b", self.session())

        captured = capsys.readouterr()

        assert "Usage: /activate <collection_name>" in captured.out

    def test_unknown_slash_command_prints_help_hint(self, capsys):
        cli.handle_user_query("/wat", "llama3.1:8b", self.session())

        captured = capsys.readouterr()

        assert "Unknown command: /wat" in captured.out
        assert "Use /help" in captured.out

    def test_empty_input_is_ignored(self, mocker):
        generate_response = mocker.patch("minirag.cli.generate_response")

        cli.handle_user_query("   ", "llama3.1:8b", self.session())

        generate_response.assert_not_called()

    def test_retrieve_without_query_prints_usage(self, capsys):
        cli.handle_user_query("/retrieve", "llama3.1:8b", self.session())

        captured = capsys.readouterr()

        assert "Usage: /retrieve <query>" in captured.out

    def test_retrieve_command_shows_retrieved_chunks(self, mocker):
        retrieve_chunks = mocker.patch("minirag.cli.retrieve_chunks_for_query")
        show_retrieved_chunks = mocker.patch("minirag.cli.show_retrieved_chunks")
        retrieve_chunks.return_value = []
        session = self.session()

        cli.handle_user_query(
            "/retrieve test question", "llama3.1:8b", session, top_k=3
        )

        retrieve_chunks.assert_called_once_with("test question", session, 3)
        show_retrieved_chunks.assert_called_once_with([])

    def test_parse_arguments_accepts_top_k(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["minirag", "--top-k", "3"])

        args = cli.parse_arguments()

        assert args.top_k == 3
