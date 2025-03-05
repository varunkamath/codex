# Codex: Code Onboarding and Documentation EXpert

Codex is a tool that allows development teams to create an "expert" AI assistant on their codebase. It ingests your codebase and documentation, allowing new developers to ask questions without constantly interrupting senior engineers.

## Features

- Ingest and process code and documentation files
- Create a searchable knowledge base from your codebase
- Ask natural language questions about your code
- Get contextually relevant answers based on your specific codebase
- Run locally or deploy to the cloud

## Installation

### Using Poetry (Recommended)

```bash
# Clone the repository
git clone https://github.com/varunkamath/codex.git
cd codex

# Install with Poetry (install Poetry first if you don't have it)
# See https://python-poetry.org/docs/#installation
poetry install

# Activate the virtual environment
poetry shell
```

### Traditional Installation

```bash
# Clone the repository
git clone https://github.com/varunkamath/codex.git
cd codex

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (compatibility method)
pip install -r requirements.txt

# Optional: Install as a package
pip install -e .
```

## Configuration

### Setting Up Environment Variables

Before using Codex, you need to set up your environment variables, particularly your OpenAI API key:

```bash
# Run the setup script
python setup_env.py
```

Alternatively, you can manually copy the `.env-example` file to `.env` and edit it:

```bash
cp .env-example .env
# Edit .env with your favorite editor
```

## Usage

### Running the Demo

The easiest way to see Codex in action is to run the demo script, which ingests the Codex codebase itself and answers questions about it:

```bash
# Using Poetry
poetry run python demo.py

# Or if you're in a poetry shell
python demo.py
```

The demo will:
1. Ingest the Codex codebase
2. Set up a query engine
3. Answer some predefined questions
4. Allow you to ask your own questions

### Ingesting a Codebase

Using Poetry:
```bash
# The correct way to run Codex on any codebase
poetry run python -m codex.main ingest --path /path/to/your/codebase --output .your_codebase_data

# Or with the simplified command (if installed as a package)
poetry run codex ingest --path /path/to/your/codebase --output .your_codebase_data
```

Or if you're in a poetry shell:
```bash
python -m codex.main ingest --path /path/to/your/codebase --output .your_codebase_data

# Or with the simplified command (if installed as a package)
codex ingest --path /path/to/your/codebase --output .your_codebase_data
```

The `--output` parameter specifies where to store the processed data, which is useful when working with multiple codebases.

### Querying the System

Using Poetry:
```bash
# The correct way to query a specific codebase
poetry run python -m codex.main query "How does the authentication system work?" --data-dir .your_codebase_data

# Or with the simplified command (if installed as a package)
poetry run codex query "How does the authentication system work?" --data-dir .your_codebase_data
```

Or if you're in a poetry shell:
```bash
python -m codex.main query "How does the authentication system work?" --data-dir .your_codebase_data

# Or with the simplified command (if installed as a package)
codex query "How does the authentication system work?" --data-dir .your_codebase_data
```

### Interactive Chat Mode

Codex now includes an interactive chat mode that keeps the model loaded between queries for faster response times:

```bash
# Start an interactive chat session with a specific codebase
poetry run python -m codex.main chat --data-dir .your_codebase_data

# With custom context window size (default is 200,000 tokens)
poetry run python -m codex.main chat --data-dir .your_codebase_data --context-window 100000

# Or with the simplified command (if installed as a package)
poetry run codex chat --data-dir .your_codebase_data
```

This mode is especially useful when you want to have a conversation about your codebase without reinitializing the model for each query. The chat maintains history between questions, providing more contextual responses.

### Working with Multiple Codebases

Codex can be used to analyze and provide assistance for different codebases. To maintain separation between them:

1. **Ingest each codebase with a unique output directory**:
   ```bash
   poetry run python -m codex.main ingest --path /path/to/codebase1 --output .codebase1_data
   poetry run python -m codex.main ingest --path /path/to/codebase2 --output .codebase2_data
   ```

2. **Query a specific codebase using its data directory**:
   ```bash
   poetry run python -m codex.main query "How does feature X work?" --data-dir .codebase1_data
   poetry run python -m codex.main query "How does feature Y work?" --data-dir .codebase2_data
   ```

3. **View statistics for a specific codebase**:
   ```bash
   poetry run python -m codex.main stats --data-dir .codebase1_data
   ```

This approach allows you to maintain separate knowledge bases for different projects.

## Architecture

Codex uses a Retrieval-Augmented Generation (RAG) approach:

1. **Ingestion**: Code and documentation are parsed, chunked, and embedded
2. **Storage**: Embeddings are stored in a vector database (ChromaDB)
3. **Retrieval**: User queries are used to find relevant code and documentation
4. **Generation**: An LLM uses the retrieved context to generate accurate answers

## Development Status

This project is currently in the prototype stage.

## License

MIT 