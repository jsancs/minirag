import ollama


def handle_model(model_name: str) -> None:
    model_exists = check_model_exists(model_name)
    if not model_exists:
        pull_model(model_name)


def check_model_exists(model_name: str) -> bool:
    all_models = ollama.list()
    ollama_model_names = [model["name"] for model in all_models["models"]]

    if model_name in ollama_model_names:
        return True
    return False


def pull_model(model_name: str) -> None:
    print(f"Model {model_name} not found. Pulling model...")
    try:
        ollama.pull(model_name)
        print("Model pulled successfully.")
    except ollama.ResponseError as e:
        print(f"Error: {e}")
        raise ValueError("Model not found. Please provide a valid model name.")