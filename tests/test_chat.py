import minirag.chat as chat
from minirag.models import ChatSession, SYS_PROMPT


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
        session = ChatSession()

        result = list(
            chat.chat_streaming("What team?", "llama3.2:1b", "CLUB: MPT", session)
        )

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
        session = ChatSession()
        session.messages.append({"role": "user", "content": "hello"})

        chat.clear_conversation(session)

        assert session.messages == [
            {"role": "system", "content": SYS_PROMPT},
        ]

    def test_build_retrieval_query_includes_recent_user_messages(self):
        session = ChatSession()
        chat.add_msg_to_memory(session, "a qué equipo pertenece el corredor?", "MPT")

        result = chat.build_retrieval_query("qué carrera es esta?", session)

        assert result == "a qué equipo pertenece el corredor?\nqué carrera es esta?"

    def test_build_retrieval_query_without_history_returns_query(self):
        session = ChatSession()

        result = chat.build_retrieval_query("qué carrera es esta?", session)

        assert result == "qué carrera es esta?"

    def test_build_retrieval_query_ignores_assistant_messages(self):
        session = ChatSession()
        chat.add_msg_to_memory(session, "a qué equipo pertenece el corredor?", "MPT")
        session.messages.append(
            {"role": "assistant", "content": "CLUB ASTORGA RUNNING"}
        )

        result = chat.build_retrieval_query("y qué carrera es?", session)

        assert result == "a qué equipo pertenece el corredor?\ny qué carrera es?"
