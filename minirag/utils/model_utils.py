from minirag.utils.backend_manager import get_backend_instance


def handle_model(model_name: str) -> None:
    backend = get_backend_instance()
    model_exists = backend.check_model_exists(model_name)
    if not model_exists:
        if backend.__class__.__name__ == "OllamaBackend":
            pull_model(model_name)
        else:
            print(
                f"Warning: Model {model_name} may not exist. Please verify the model name."
            )


def check_model_exists(model_name: str) -> bool:
    backend = get_backend_instance()
    return backend.check_model_exists(model_name)


def pull_model(model_name: str) -> None:
    backend = get_backend_instance()
    backend.pull_model(model_name)
