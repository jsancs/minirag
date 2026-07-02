from typing import Generator

from minirag.utils.backend_manager import get_backend_instance

SYS_PROMPT = (
    "You are a knowledgeable, efficient, and direct AI assistant. Provide concise "
    "answers, focusing on the key information needed. When context is provided, "
    "answer using only that context. If the answer is not in the context, say you "
    "do not know. The current context has priority over the conversation history. "
    "In forms, distinguish participant or registration fields from event and "
    "organizer metadata. Use organizer metadata only when the user asks about the "
    "organizer."
)
CONVERSATION_HISTORY = [
    {"role": "system", "content": SYS_PROMPT},
]
RETRIEVAL_HISTORY_LIMIT = 3


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


def build_retrieval_query(query: str) -> str:
    previous_user_messages = [
        message["content"]
        for message in CONVERSATION_HISTORY
        if message["role"] == "user"
    ][-RETRIEVAL_HISTORY_LIMIT:]

    if not previous_user_messages:
        return query

    return "\n".join([*previous_user_messages, query])


def chat_streaming(
    query: str,
    model: str,
    context: str | None = None,
) -> Generator[str, None, None]:
    backend = get_backend_instance()
    stream = backend.chat_streaming(
        model=model,
        messages=[
            *CONVERSATION_HISTORY,
            {
                "role": "user",
                "content": build_user_message(query, context),
            },
        ],
    )

    for chunk in stream:
        yield chunk


def add_msg_to_memory(user_query: str, model_response: str) -> None:
    CONVERSATION_HISTORY.extend(
        [
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": model_response},
        ]
    )


def clear_conversation() -> None:
    global CONVERSATION_HISTORY
    CONVERSATION_HISTORY = [
        {"role": "system", "content": SYS_PROMPT},
    ]
