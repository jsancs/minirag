from dataclasses import dataclass, field

SYS_PROMPT = (
    "You are a knowledgeable, efficient, and direct AI assistant. Provide concise "
    "answers, focusing on the key information needed. When context is provided, "
    "answer using only that context. If the answer is not in the context, say you "
    "do not know. The current context has priority over the conversation history. "
    "In forms, distinguish participant or registration fields from event and "
    "organizer metadata. Use organizer metadata only when the user asks about the "
    "organizer."
)
RETRIEVAL_HISTORY_LIMIT = 3


def build_initial_messages() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYS_PROMPT},
    ]


@dataclass
class ChatSession:
    messages: list[dict[str, str]] = field(default_factory=build_initial_messages)

    def add_exchange(self, user_query: str, model_response: str) -> None:
        self.messages.extend(
            [
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": model_response},
            ]
        )

    def clear(self) -> None:
        self.messages = build_initial_messages()

    def build_retrieval_query(self, query: str) -> str:
        previous_user_messages = [
            message["content"] for message in self.messages if message["role"] == "user"
        ][-RETRIEVAL_HISTORY_LIMIT:]

        if not previous_user_messages:
            return query

        return "\n".join([*previous_user_messages, query])
