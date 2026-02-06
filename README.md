# mini-local-rag

A tiny implementation of a RAG system that runs entirely on your computer!

> [!IMPORTANT]  
> This repo is in development. It can have important changes between commits

## Start

1. Install dependencies with uv:
   ```bash
   uv sync
   ```
2. Run the CLI:
   - For Ollama (default): `uv run minirag`
   - For OpenAI: `uv sync --extra openai && uv run minirag --backend openai --model gpt-4`
3. Chat with the model

There are 2 configurable params: <br>
* `-m --model`: model to use (llama3.2:1b by default for Ollama). You can check the full list of available models [here](https://ollama.com/library)
* `-b --backend`: backend to use (ollama by default). Options: ollama, openai. OpenAI backend requires the optional dependency.


## Usage
* Type a message to chat with the model. All the conversation will be remembered by the model.
* Type `/bye` to exit the chat.
* Type `/help` to show all the commands.
* Type `/add` to create a collection.
    * You'll be asked to enter the paths for all the documents for the collection. You can enter specific files or directories, in which case it will process all the files within the directory.
    * You will be asked to introduce a collection name.
    * Then the embeddings will be generated and stored in a .npy file for future reference. The embeddings will be stored in memory with numpy.
* Type `/activate` to load and use a collection.
* Type `/deactivate` to deactivate the active collection.
* Type `/list` to list available collections.


## Backends

mini-local-rag supports multiple backends for AI model inference:

### Ollama (default)
- Local inference, no API keys required
- Install: Start Ollama engine
- Usage: `uv run minirag` or `uv run minirag --backend ollama`

### OpenAI
- Cloud-based OpenAI API (and OpenAI-compatible endpoints like vllm, openrouter)
- Requires `OPENAI_API_KEY` environment variable
- Optional `OPENAI_BASE_URL` for compatible endpoints (default: https://api.openai.com/v1)
- Models are cloud-hosted and don't need to be pulled
- Install: `uv sync --extra openai`
- Usage: `uv run minirag --backend openai --model gpt-4`

### Adding a new backend
To add a custom backend, create a new class in `minirag/backends/` inheriting from `Backend` and implement the abstract methods. Register it in `minirag/backends/__init__.py`.


## Roadmap
These are the next steps I plan to take:

- [ ] Support vision models
- [ ] Support for more files (see section below)
- [X] Testing
- [ ] Improve index algorithm
- [ ] Performance metrics (speed, storage, scalability, ...)
- [ ] UI (somthing very light and simple)

## Project Setup

This project uses uv for dependency management. The project configuration is in `pyproject.toml`:

- Core dependencies are listed under `[project.dependencies]`
- Development dependencies are listed under `[tool.uv.dev-dependencies]`
- To install dependencies, run `uv sync`
- To run the CLI, use `uv run minirag` or `uv run minirag --model <model_name>`
- To run tests, use `uv run pytest tests`

Feel free to suggest any other relevant topic or idea to be included in the code (contributions are also welcome)


## Supported files

* .txt
* .pdf


## Testing
Testing has been done with pytest.

To run all the tests:
`uv run pytest tests`

To generate coverage report:
`uv run pytest --cov-report term --cov-report xml:coverage.xml --cov=minirag`


## Evaluation (in progress)
The `evaluation/` folder contains scripts and data used for evaluate the performance of this rag system.
The supported metrics right now are the following:

### Similarity search speed
This system uses the basic `np.dot` function to compute the similarity search between the embeddings. To compute the eval data for this metric execute the following command: `python -m evaluation.eval`, it will generate a `.csv` file with the result benchmark for different collection sizes.