# mini-local-rag

A tiny implementation of a RAG system that runs entirely on your computer!

> [!IMPORTANT]  
> This repo is in development. It can have important changes between commits

## Start

1. Start Ollama engine.
2. Install dependencies with uv:
   ```bash
   uv sync
   ```
3. Run `uv run minirag`. It will run a Llama3.2:1b model.
4. Chat with the model

There is 1 configurable param: <br>
* `-m --model`: model to use (llama3.2:1b by default). You can check the full list of available models [here](https://ollama.com/library)


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