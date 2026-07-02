from typing import Generator

from minirag.models import ChatSession
from minirag.utils.backend_manager import get_backend_instance


def build_user_message(query: str, context: str | None = None) -> str:
    if context is None:
        return query

    return (
        "Context:\n"
        f"{context}\n\n"
        "Instructions:\n"
        "Use the field that directly answers the question. In forms, distinguish "
        "participant or registration data from event and organizer metadata. "
        "Use organizer metadata only when the user asks about the organizer. "
        "Answer briefly.\n\n"
        "Question:\n"
        f"{query}\n\n"
        "Answer:"
    )


def chat_streaming(
    query: str,
    model: str,
    context: str | None = None,
    session: ChatSession | None = None,
) -> Generator[str, None, None]:
    active_session = session or ChatSession()
    backend = get_backend_instance()
    stream = backend.chat_streaming(
        model=model,
        messages=[
            *active_session.messages,
            {
                "role": "user",
                "content": build_user_message(query, context),
            },
        ],
    )

    for chunk in stream:
        yield chunk


def add_msg_to_memory(
    session: ChatSession,
    user_query: str,
    model_response: str,
) -> None:
    session.add_exchange(user_query, model_response)


def clear_conversation(session: ChatSession) -> None:
    session.clear()


def build_retrieval_query(query: str, session: ChatSession) -> str:
    return session.build_retrieval_query(query)
