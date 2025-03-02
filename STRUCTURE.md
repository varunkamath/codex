# Codex Project Structure

This document provides an overview of the Codex codebase organization, to help new developers understand how the system is structured.

## Directory Structure

```
codex/
├── .env-example         # Template for environment variables
├── .gitignore           # Git ignore patterns
├── LICENSE              # Project license
├── README.md            # Project documentation
├── STRUCTURE.md         # This file
├── demo.py              # Demo script to showcase functionality
├── poetry.lock          # Poetry lock file (dependencies)
├── pyproject.toml       # Poetry project configuration
├── requirements.txt     # Pip requirements (compatibility)
├── setup_env.py         # Environment setup script
├── tests/               # Test directory
│   └── __init__.py      # Package initialization for tests
└── codex/               # Main package directory
    ├── __init__.py      # Package initialization
    ├── main.py          # Command-line entry point
    ├── ingestion/       # Code and doc ingestion components
    │   ├── __init__.py
    │   ├── code_parser.py   # Code parsing functionality
    │   └── doc_parser.py    # Documentation parsing
    ├── storage/         # Storage components
    │   ├── __init__.py
    │   └── vector_store.py  # Vector database management
    ├── llm/             # Language model components
    │   ├── __init__.py
    │   └── query_engine.py  # LLM query processing
    └── ui/              # User interface components
        ├── __init__.py
        └── cli.py           # Command-line interface
```

## Core Components

### Ingestion Components

- **`ingestion/code_parser.py`**: Handles parsing of code files in various languages.
  - Class: `CodeParser`
  - Methods: 
    - `find_code_files()`: Discovers code files in the repository
    - `parse_file()`: Extracts content from a code file
    - `chunk_code()`: Breaks code into chunks for embedding
    - `process_codebase()`: End-to-end processing of a codebase

- **`ingestion/doc_parser.py`**: Processes documentation files (Markdown, HTML, PDF, etc.).
  - Class: `DocParser`
  - Methods:
    - `find_doc_files()`: Discovers documentation files
    - `parse_file()`: Extracts content from documentation files
    - `chunk_text()`: Breaks text into chunks for embedding
    - `process_docs()`: End-to-end processing of documentation

### Storage Component

- **`storage/vector_store.py`**: Manages the vector database (ChromaDB) for storing and retrieving embedded chunks.
  - Class: `VectorStore`
  - Methods: 
    - `add_code_chunks()`: Adds code chunks to the vector store
    - `add_doc_chunks()`: Adds documentation chunks to the vector store
    - `search_code()`: Searches for relevant code chunks
    - `search_docs()`: Searches for relevant documentation chunks
    - `search_all()`: Searches across all content types
    - `get_stats()`: Returns statistics about the vector store

### Query Component

- **`llm/query_engine.py`**: Handles user queries, retrieves relevant chunks, and generates responses using the language model.
  - Class: `QueryEngine`
  - Methods: 
    - `query()`: Process a user query and generate a response
    - `get_stats()`: Returns information about queries processed

### User Interface

- **`ui/cli.py`**: Command-line interface for interacting with Codex.
  - Commands: `ingest`, `query`, `stats`

### Entry Points

- **`main.py`**: Command-line interface entry point.

- **`demo.py`**: Interactive demo that showcases the system.

## Data Flow

1. **Ingestion Process**:
   - `CodeParser` discovers code files and processes them into chunks
   - `DocParser` discovers documentation files and processes them into chunks
   - `VectorStore` embeds these chunks and stores them in ChromaDB

2. **Query Process**:
   - User query is processed by `QueryEngine`
   - `QueryEngine` uses `VectorStore` to retrieve relevant chunks
   - Language model generates a response based on the context
   - Response is formatted and presented to the user

## Configuration

- **Environment Variables**: Stored in `.env` file
  - `OPENAI_API_KEY`: API key for OpenAI
  - `OPENAI_MODEL_NAME`: Model to use (default: gpt-3.5-turbo)
  - `PERSIST_DIRECTORY`: Where to store the vector database
  - `LOCAL_MODEL_PATH`: Path to local model weights (optional)

## Testing

Tests are located in the `tests` directory and can be run with:

```bash
poetry run pytest
```

## Extending Codex

To add support for new file types:
1. Update the appropriate parser class in the ingestion module
2. Add file extension detection logic
3. Implement parsing functionality for the new format

To add new features:
1. Identify the component that should be extended
2. Create new methods or update existing ones
3. Update the CLI interface in `ui/cli.py` if needed
4. Add tests for new functionality 