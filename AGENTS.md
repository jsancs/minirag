# AGENTS.md

## Build, Lint, and Test Commands

Run all tests:
```bash
python -m pytest tests
```

Run a single test:
```bash
python -m pytest tests/services/test_rag_service.py::TestRagService::test_get_splitter
```

Generate coverage report:
```bash
python -m pytest --cov-report term --cov-report xml:coverage.xml --cov=minirag
```

## Code Style Guidelines

### File Structure
- Main CLI entry point: `minirag/cli.py`
- Services layer: `minirag/services/`
- Models: `minirag/models/`
- Utilities: `minirag/utils/`
- Tests: `tests/`

### Import Organization
1. Standard library imports (typing, os, etc.)
2. Third-party imports (ollama, numpy, langchain, etc.)
3. Local module imports (from minirag import ...)

Example:
```python
from typing import List, Optional
import ollama
import numpy as np
from minirag.models import Chunk
```

### Naming Conventions
- **Modules**: lowercase with underscores (snake_case)
- **Classes**: PascalCase (e.g., `RagService`, `CollectionService`)
- **Functions/Methods**: snake_case (e.g., `generate_embeddings`, `load_collection`)
- **Constants**: UPPER_SNAKE_CASE
- **Private methods**: single underscore prefix (e.g., `_process_folder`)

### Code Formatting
- Use 4 spaces for indentation (no tabs)
- Function definitions: open parentheses on same line
- Imports grouped by source (standard, third-party, local)
- Line length: max 100 characters

### Type Hints
- Use type annotations for all function parameters and return values
- Use `Optional[T]` for nullable values
- Use built-in types (List, Dict, str, int, float, etc.)
- Use `Generator[str, None, None]` for generators

Example:
```python
def similarity_search(
    query: str,
    collection: List[Chunk],
    top_k: int = 5,
) -> str:
    ...
```

### Data Structures
- Use `dataclass` for simple data models
- Add type hints to dataclass fields

Example:
```python
from dataclasses import dataclass

@dataclass
class Chunk:
    document_name: str
    text: str
    embedding: list[float]
```

### Error Handling
- Use try/except for external operations (file I/O, HTTP requests)
- Print errors to console for user feedback
- Return empty strings or None on failure
- Raise exceptions for critical failures

Example:
```python
try:
    with open(doc_path, "r") as f:
        text = f.read()
except FileNotFoundError:
    print(f"File not found: {doc_path}")
    return ""
```

### Static Methods
- Use `@staticmethod` for utility methods in services
- No `self` parameter

Example:
```python
@staticmethod
def get_splitter(chunk_size: int = 1000, chunk_overlap: int = 20) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
```

### Testing
- Use pytest for all tests
- Mock external dependencies with pytest-mock (`mocker`)
- Use descriptive test class names (`TestRagService`)
- Use descriptive test method names (`test_get_splitter`)
- Test both default and custom parameters
- Verify mocked functions are called correctly

Example:
```python
def test_generate_embeddings(self, mocker):
    mock_embeddings = mocker.patch('ollama.embeddings')
    mock_embeddings.return_value = {"embedding": [0.1, 0.2, 0.3]}

    embeddings = RagService.generate_embeddings("test")
    assert embeddings == [0.1, 0.2, 0.3]
    mock_embeddings.assert_called_once_with(model="all-minilm", prompt="test")
```

### Logging
- Use simple print statements for user-facing output
- Use `print()` with clear messages
- Include function names in error messages

### Class Design
- Keep services as singletons (module-level instances)
- Use descriptive method names
- Group related functionality in service classes

### Documentation
- Minimal docstrings (only for complex logic)
- Type hints serve as primary documentation
- Test functions document expected behavior

### Code Organization
- Keep methods small and focused
- Avoid deep nesting (max 2-3 levels)
- Use early returns for simple conditions
- Group related code logically

### Error Messages
- Use clear, user-friendly messages
- Include relevant context (file paths, function names)
- End with period for sentences

### Performance Considerations
- Use numpy for vector operations
- Cache expensive computations when possible
- Use generator patterns for streaming responses
