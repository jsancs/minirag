import argparse
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from prompt_toolkit.shortcuts import prompt

from minirag.chat import (
    add_msg_to_memory,
    build_retrieval_query,
    chat_streaming,
    clear_conversation,
)
from minirag.models import ChatSession, Chunk
from minirag.services.collection_service import CollectionService
from minirag.services.rag_service import RagService
from minirag.utils.model_utils import handle_model
from minirag.utils.backend_manager import set_backend
from minirag.backends import BACKENDS

collection_service = CollectionService()
DEFAULT_TOP_K = 5


def show_help() -> None:
    print("MiniRAG commands:")
    print("  /add                         Create a collection from files or folders.")
    print("  /activate <collection_name>  Load a collection for RAG answers.")
    print("  /deactivate                  Unload the active collection.")
    print(
        "  /retrieve <query>            Show retrieved chunks without asking the model."
    )
    print("  /list                        Show saved collections.")
    print("  /status                      Show the active collection state.")
    print("  /clear                       Clear conversation history and the terminal.")
    print("  /help, /?                    Show this help message.")
    print("  /bye, /exit, Ctrl-D          Exit the chat.")
    print()
    print("Shortcuts:")
    print("  Ctrl-C                       Stop the current response.")
    print()
    print("Options:")
    print("  --top-k                      Number of chunks to retrieve for RAG.")
    print()


def clear_terminal() -> None:
    subprocess.run("cls" if os.name == "nt" else "clear", shell=True, check=False)


def normalize_doc_path(doc_path: str) -> str:
    doc_path = doc_path.strip()
    if len(doc_path) >= 2 and doc_path[0] == doc_path[-1] and doc_path[0] in ("'", '"'):
        return doc_path[1:-1]
    return doc_path


def get_user_input() -> str:
    user_query = prompt(
        ">>> ",
        placeholder="Send a message (/? for help)",
    )
    return user_query


def generate_response(
    user_query: str,
    model_name: str,
    session: ChatSession,
    context: str | None = None,
) -> None:
    model_response = ""

    for chunk in chat_streaming(user_query, model_name, context, session):
        model_response += chunk
        print(chunk, end="", flush=True)
    print()

    add_msg_to_memory(session, user_query, model_response)


def chunks_to_context(chunks: list[Chunk]) -> str:
    return "\n\n".join([chunk.text for chunk in chunks])


def format_chunk_source(chunk: Chunk) -> str:
    source = Path(chunk.document_name).name
    metadata = [source, f"chunk {chunk.chunk_index}"]

    if chunk.page_number is not None:
        metadata.insert(1, f"page {chunk.page_number}")

    return ", ".join(metadata)


def show_retrieved_chunks(chunks: list[Chunk]) -> None:
    if not chunks:
        print("No chunks retrieved.")
        return

    for result_index, chunk in enumerate(chunks, start=1):
        print(f"[{result_index}] {format_chunk_source(chunk)}")
        print(f"score: {chunk.similarity:.4f}")
        if chunk.chunk_id:
            print(f"id: {chunk.chunk_id}")
        print(chunk.text)
        print()


def get_documents() -> list[str]:
    doc_paths = []
    while True:
        doc_path = prompt(
            "Local path to document (file or dir) (/done to finish): ",
        )
        if doc_path == "/done":
            break

        if doc_path:
            doc_paths.append(normalize_doc_path(doc_path))

    return doc_paths


def show_status() -> None:
    if collection_service.active_collection:
        chunk_count = len(collection_service.active_collection)
        print(f"Collection active ({chunk_count} chunks).")
    else:
        print("No collection active.")


def retrieve_chunks_for_query(
    user_query: str,
    session: ChatSession,
    top_k: int = DEFAULT_TOP_K,
) -> list[Chunk]:
    if not collection_service.active_collection:
        print("No collection active. Use /activate <collection_name> first.")
        return []

    retrieval_query = build_retrieval_query(user_query, session)
    return RagService.retrieve_chunks(
        retrieval_query,
        collection_service.active_collection,
        top_k=top_k,
    )


def handle_user_query(
    user_query: str,
    model_name: str,
    session: ChatSession,
    top_k: int = DEFAULT_TOP_K,
) -> None:
    user_query = user_query.strip()

    if not user_query:
        return

    if user_query in ("/bye", "/exit"):
        print("Goodbye!")
        exit()
    elif user_query == "/clear":
        clear_conversation(session)
        clear_terminal()
        print("Conversation cleared.")
    elif user_query == "/help" or user_query == "/?":
        show_help()
    elif user_query.startswith("/activate"):
        command_parts = user_query.split(maxsplit=1)
        if len(command_parts) == 1:
            print("Usage: /activate <collection_name>")
            return

        collection_name = command_parts[1]
        collection_service.load_collection(collection_name)
    elif user_query == "/list":
        collection_service.list_collections()
    elif user_query == "/status":
        show_status()
    elif user_query == "/deactivate":
        print("Collection deactivated.")
        collection_service.active_collection = None
    elif user_query.startswith("/retrieve"):
        command_parts = user_query.split(maxsplit=1)
        if len(command_parts) == 1:
            print("Usage: /retrieve <query>")
            return

        retrieved_chunks = retrieve_chunks_for_query(command_parts[1], session, top_k)
        show_retrieved_chunks(retrieved_chunks)
    elif user_query == "/add":
        collection_name = ""
        documents = get_documents()
        if documents:
            collection_name = prompt(
                "Enter a name   for the collection: ",
            )

        print(f"Adding documents to collection: {collection_name}")
        print(f"Docs selected: {documents}")

        collection_service.create_collection(documents, collection_name)
    elif user_query.startswith("/"):
        print(f"Unknown command: {user_query}")
        print("Use /help to see available commands.")
    else:
        if collection_service.active_collection:
            retrieved_chunks = retrieve_chunks_for_query(user_query, session, top_k)
            context = chunks_to_context(retrieved_chunks)
        else:
            context = None

        generate_response(user_query, model_name, session, context)


def chat_cli(model_name: str, top_k: int = DEFAULT_TOP_K) -> None:
    session = ChatSession()
    while True:
        try:
            user_query = get_user_input()
            handle_user_query(user_query, model_name, session, top_k)

        except KeyboardInterrupt:
            # Ctrl-C to stop the model from responding
            print("\nUse Ctrl + d or /bye to exit.")
        except EOFError:
            # Ctrl-D to exit
            break


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with an AI assistant.")
    parser.add_argument(
        "-m", "--model", type=str, default="llama3.1:8b", help="Model to use."
    )
    parser.add_argument(
        "-b",
        "--backend",
        type=str,
        default="ollama",
        choices=list(BACKENDS.keys()),
        help="Backend to use (ollama, openai).",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=parse_positive_int,
        default=DEFAULT_TOP_K,
        help="Number of chunks to retrieve for RAG.",
    )
    return parser.parse_args()


def parse_positive_int(value: str) -> int:
    parsed_value = int(value)
    if parsed_value < 1:
        raise argparse.ArgumentTypeError("Value must be greater than 0.")

    return parsed_value


def main() -> None:
    load_dotenv()
    args = parse_arguments()
    model_name = args.model
    backend_name = args.backend
    top_k = args.top_k

    set_backend(backend_name)

    handle_model(model_name)
    chat_cli(model_name, top_k)


if __name__ == "__main__":
    main()
