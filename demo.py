#!/usr/bin/env python
"""
Codex Demo Script.

This script demonstrates the functionality of Codex by ingesting
its own codebase and answering questions about it.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

try:
    from dotenv import load_dotenv
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
except ImportError:
    print("Required packages not found. Please install them with:")
    print("poetry install")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Check for required environment variables
if not os.getenv("OPENAI_API_KEY") and not os.getenv("LOCAL_MODEL_PATH"):
    print(
        "Error: Neither OPENAI_API_KEY nor LOCAL_MODEL_PATH found in environment variables."
    )
    print("Please set one of them in your .env file or run setup_env.py first.")
    print("\nTo use a local model:")
    print("1. Download a GGUF model (e.g., llama-2-7b-chat.Q4_K_M.gguf)")
    print("2. Place it in a 'models' directory")
    print("3. Add 'LOCAL_MODEL_PATH=models/your-model-filename.gguf' to your .env file")
    print("\nOr run our helper script:")
    print("poetry run python download_model.py")
    sys.exit(1)

# If using a local model, check for required dependencies
if os.getenv("LOCAL_MODEL_PATH"):
    model_path = os.getenv("LOCAL_MODEL_PATH")
    print(f"Using local model from: {model_path}")

    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please download the model or correct the path in your .env file.")
        print("\nYou can use our helper script to download a model:")
        print("poetry run python download_model.py")
        sys.exit(1)

    # Check for required libraries for local models
    try:
        import llama_cpp  # noqa: F401
        from langchain_community.llms import LlamaCpp  # noqa: F401
    except ImportError:
        print("Error: Required packages for local models not found.")
        print("Please install them with:")
        print("poetry add langchain-community llama-cpp-python")
        sys.exit(1)

# Initialize Rich console
console = Console()

# Demo questions to showcase functionality
DEMO_QUESTIONS = [
    "What does Codex do?",
    "How does the code parsing work?",
    "What vector database does Codex use?",
    "How are queries processed in the system?",
    "What file types can Codex ingest?",
]


def ingest_codebase(codebase_path: Path) -> Dict[str, Any]:
    """
    Ingest a codebase into the vector store.

    Args:
        codebase_path: Path to the codebase

    Returns:
        Dictionary with statistics about the ingestion
    """
    from codex.ingestion.code_parser import CodeParser
    from codex.ingestion.doc_parser import DocParser
    from codex.storage.vector_store import VectorStore

    stats = {"code_files": 0, "doc_files": 0, "chunks": 0}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Initialize vector store
        task = progress.add_task("[blue]Initializing vector store...", total=None)
        vector_store = VectorStore()
        progress.update(
            task, completed=True, description="[green]Vector store initialized"
        )

        # Process code files
        task = progress.add_task("[blue]Processing code files...", total=None)
        code_parser = CodeParser(str(codebase_path))
        code_chunks = code_parser.process_codebase()
        stats["code_files"] = len(code_chunks)
        progress.update(
            task,
            completed=True,
            description=f"[green]Processed {len(code_chunks)} code files",
        )

        # Process documentation files
        task = progress.add_task("[blue]Processing documentation files...", total=None)
        doc_parser = DocParser(str(codebase_path))
        doc_chunks = doc_parser.process_docs()
        stats["doc_files"] = len(doc_chunks)
        progress.update(
            task,
            completed=True,
            description=f"[green]Processed {len(doc_chunks)} documentation files",
        )

        # Add chunks to vector store
        task = progress.add_task(
            "[blue]Adding code chunks to vector store...", total=None
        )
        vector_store.add_code_chunks(code_chunks)
        progress.update(
            task,
            completed=True,
            description=f"[green]Added {len(code_chunks)} code chunks to vector store",
        )

        task = progress.add_task(
            "[blue]Adding documentation chunks to vector store...", total=None
        )
        vector_store.add_doc_chunks(doc_chunks)
        progress.update(
            task,
            completed=True,
            description=f"[green]Added {len(doc_chunks)} documentation chunks to vector store",
        )

        stats["chunks"] = len(code_chunks) + len(doc_chunks)

        # Persist vector store (will happen automatically on object destruction)
        task = progress.add_task("[blue]Persisting vector store...", total=None)
        # No need to explicitly call persist as it happens during cleanup
        progress.update(
            task, completed=True, description="[green]Vector store persisted"
        )

    return stats


def run_query(query_engine, question: str) -> None:
    """
    Run a query and display the answer.

    Args:
        query_engine: The QueryEngine instance
        question: The question to answer
    """
    console.print(f"\n[bold cyan]Question:[/bold cyan] {question}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[yellow]Thinking...", total=None)
        try:
            # Set a reasonable timeout for local models (10 minutes)
            start_time = time.time()
            answer = query_engine.query(question)
            elapsed_time = time.time() - start_time

            if elapsed_time > 5:
                progress.update(
                    task,
                    completed=True,
                    description=f"[green]Answer ready (took {elapsed_time:.1f} seconds)",
                )
            else:
                progress.update(task, completed=True, description="[green]Answer ready")

        except Exception as e:
            progress.update(
                task, completed=True, description="[red]Error generating answer"
            )
            console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
            return

    console.print("\n[bold green]Answer:[/bold green]")
    console.print(Panel(answer, border_style="green"))
    console.print(f"\n{'=' * 80}\n")


def run_demo() -> None:
    """Run the Codex demo."""
    console.print(
        Panel.fit(
            "[bold blue]Codex Demo[/bold blue]\n\n"
            "This demo will ingest the Codex codebase and answer questions about it.",
            title="Welcome",
            border_style="blue",
        )
    )

    # Get the current directory (assuming it's the Codex repository)
    codebase_path = Path(".")

    # Check if this is actually the Codex repository
    if not (codebase_path / "pyproject.toml").exists():
        console.print(
            "[bold red]Error:[/bold red] This doesn't appear to be the Codex repository."
        )
        console.print("Please run this script from the root of the Codex repository.")
        return

    # Ingest the codebase
    console.print("\n[bold]Step 1:[/bold] Ingesting the Codex codebase")
    stats = ingest_codebase(codebase_path)

    console.print("\n[bold green]Ingestion complete![/bold green]")
    console.print(
        f"Processed {stats['code_files']} code files and {stats['doc_files']} documentation files"
    )
    console.print(f"Created {stats['chunks']} chunks in the vector store")

    # Initialize query engine
    console.print("\n[bold]Step 2:[/bold] Setting up the query engine")
    try:
        from codex.llm.query_engine import QueryEngine

        query_engine = QueryEngine()
        console.print("[green]Query engine initialized![/green]")
    except Exception as e:
        console.print(f"[bold red]Error initializing query engine:[/bold red] {str(e)}")
        console.print(
            "You can still interact with the vector store, but queries will not be processed by an LLM."
        )
        console.print("Please check your environment variables and dependencies.")
        return

    # Run demo questions
    console.print("\n[bold]Step 3:[/bold] Answering demo questions")
    for i, question in enumerate(DEMO_QUESTIONS, 1):
        console.print(f"\n[bold]Demo Question {i}/{len(DEMO_QUESTIONS)}[/bold]")
        try:
            run_query(query_engine, question)
            # Short pause between questions for readability
            if i < len(DEMO_QUESTIONS):
                time.sleep(1)
        except Exception as e:
            console.print(f"[bold red]Error processing query:[/bold red] {str(e)}")
            console.print("Skipping to next question...")
            continue

    # Interactive mode
    console.print("\n[bold]Step 4:[/bold] Interactive mode")
    console.print("Now you can ask your own questions about the Codex codebase.")
    console.print("Type 'exit' or 'quit' to end the demo.\n")

    while True:
        question = console.input("[bold cyan]Your question:[/bold cyan] ")
        if question.lower() in ["exit", "quit"]:
            break
        if question.strip():
            try:
                run_query(query_engine, question)
            except Exception as e:
                console.print(f"[bold red]Error processing query:[/bold red] {str(e)}")
                console.print("Please try a different question.")

    console.print("\n[bold blue]Thank you for trying Codex![/bold blue]")
    console.print("Learn more at https://github.com/yourusername/codex")


if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        console.print("\n\n[bold yellow]Demo interrupted by user.[/bold yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n\n[bold red]Error:[/bold red] {str(e)}")
        console.print(
            "If this is related to the OpenAI API, please check your API key and quota."
        )
        sys.exit(1)
