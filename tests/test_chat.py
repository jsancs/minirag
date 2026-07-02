import minirag.chat as chat


class TestChat:
    def test_build_user_message_without_context(self):
        assert chat.build_user_message("What team is the runner on?") == (
            "What team is the runner on?"
        )

    def test_build_user_message_with_context(self):
        result = chat.build_user_message(
            "What team is the runner on?",
            "NAME: Javier\nCLUB: MPT",
        )

        assert "Context:\nNAME: Javier\nCLUB: MPT" in result
        assert "distinguish participant or registration data" in result
        assert "Question:\nWhat team is the runner on?" in result
        assert result.endswith("Answer:")

    def test_chat_streaming_sends_context_as_user_message(self, mocker):
        mock_backend = mocker.MagicMock()
        mock_backend.chat_streaming.return_value = iter(["MPT"])
        mocker.patch("minirag.chat.get_backend_instance", return_value=mock_backend)

        result = list(chat.chat_streaming("What team?", "llama3.2:1b", "CLUB: MPT"))

        assert result == ["MPT"]
        mock_backend.chat_streaming.assert_called_once()
        call_kwargs = mock_backend.chat_streaming.call_args.kwargs
        assert call_kwargs["model"] == "llama3.2:1b"
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][-1] == {
            "role": "user",
            "content": (
                "Context:\n"
                "CLUB: MPT\n\n"
                "Instructions:\n"
                "Use the field that directly answers the question. In forms, distinguish "
                "participant or registration data from event and organizer metadata. "
                "Use organizer metadata only when the user asks about the organizer. "
                "Answer briefly.\n\n"
                "Question:\n"
                "What team?\n\n"
                "Answer:"
            ),
        }

    def test_clear_conversation_keeps_system_prompt(self):
        chat.CONVERSATION_HISTORY.append({"role": "user", "content": "hello"})

        chat.clear_conversation()

        assert chat.CONVERSATION_HISTORY == [
            {"role": "system", "content": chat.SYS_PROMPT},
        ]

    def test_build_retrieval_query_includes_recent_user_messages(self):
        chat.clear_conversation()
        chat.add_msg_to_memory("a qué equipo pertenece el corredor?", "MPT")

        result = chat.build_retrieval_query("qué carrera es esta?")

        assert result == "a qué equipo pertenece el corredor?\nqué carrera es esta?"

    def test_build_retrieval_query_without_history_returns_query(self):
        chat.clear_conversation()

        result = chat.build_retrieval_query("qué carrera es esta?")

        assert result == "qué carrera es esta?"
