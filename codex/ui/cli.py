"""
Command Line Interface - CLI for Codex.
"""

import os
import time
import logging
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from ..ingestion.code_parser import CodeParser
from ..ingestion.doc_parser import DocParser
from ..storage.vector_store import VectorStore
from ..llm.query_engine import QueryEngine

# Set up console for rich output
console = Console()


def setup_logger():
    """Set up logging for the application."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=log_format, handlers=[logging.StreamHandler()]
    )
    # Reduce verbosity of some noisy libraries
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)


def show_header():
    """Show application header."""
    console.print(
        Panel.fit(
            "[bold blue]Codex[/bold blue]: [cyan]Code Onboarding and Documentation EXpert[/cyan]",
            border_style="blue",
        )
    )


def ingest_codebase(
    path: str,
    output_dir: str = ".codex_data",
    code_chunk_size: int = 1000,
    doc_chunk_size: int = 1500,
):
    """
    Ingest a codebase into the vector store.

    Args:
        path: Path to the codebase directory
        output_dir: Directory to store the processed data
        code_chunk_size: Maximum chunk size for code files
        doc_chunk_size: Maximum chunk size for documentation files
    """
    show_header()
    console.print(f"[bold green]Ingesting codebase[/bold green]: {path}")
    console.print(f"[bold]Output directory[/bold]: {output_dir}")

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    # Set up vector store
    vector_store = VectorStore(output_dir)

    # Ingest code files
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # Parse code files
        task_id = progress.add_task("[bold blue]Parsing code files...", total=None)
        code_parser = CodeParser(path)
        code_files = code_parser.find_code_files()
        progress.update(task_id, total=len(code_files))
        progress.update(
            task_id,
            description=f"[bold blue]Processing {len(code_files)} code files...",
        )

        # Process code files
        all_code_chunks = []
        for i, (file_path, language) in enumerate(code_files):
            progress.update(
                task_id,
                advance=1,
                description=f"[bold blue]Processing {i + 1}/{len(code_files)} code files...",
            )
            file_data = code_parser.parse_file(file_path, language)
            if "error" not in file_data:
                chunks = code_parser.chunk_code(
                    file_data["content"], language, code_chunk_size
                )
                for j, chunk in enumerate(chunks):
                    chunk["file_path"] = file_data["path"]
                    chunk["chunk_index"] = j
                    chunk["total_chunks"] = len(chunks)
                all_code_chunks.extend(chunks)

        # Add code chunks to vector store
        progress.update(
            task_id,
            total=None,
            description="[bold blue]Adding code chunks to vector store...",
        )
        vector_store.add_code_chunks(all_code_chunks)
        progress.update(
            task_id,
            description=f"[bold blue]Added {len(all_code_chunks)} code chunks to vector store",
        )

        # Parse documentation files
        task_id = progress.add_task(
            "[bold green]Parsing documentation files...", total=None
        )
        doc_parser = DocParser(path)
        doc_files = doc_parser.find_doc_files()
        progress.update(task_id, total=len(doc_files))
        progress.update(
            task_id,
            description=f"[bold green]Processing {len(doc_files)} documentation files...",
        )

        # Process documentation files
        all_doc_chunks = []
        for i, doc_file in enumerate(doc_files):
            progress.update(
                task_id,
                advance=1,
                description=f"[bold green]Processing {i + 1}/{len(doc_files)} documentation files...",
            )
            file_data = doc_parser.parse_file(doc_file)
            if "error" not in file_data:
                chunks = doc_parser.chunk_text(file_data["content"], doc_chunk_size)
                for j, chunk in enumerate(chunks):
                    chunk["file_path"] = file_data["path"]
                    chunk["format"] = file_data["format"]
                    chunk["chunk_index"] = j
                    chunk["total_chunks"] = len(chunks)
                all_doc_chunks.extend(chunks)

        # Add documentation chunks to vector store
        progress.update(
            task_id,
            total=None,
            description="[bold green]Adding documentation chunks to vector store...",
        )
        vector_store.add_doc_chunks(all_doc_chunks)
        progress.update(
            task_id,
            description=f"[bold green]Added {len(all_doc_chunks)} documentation chunks to vector store",
        )

    # Show summary
    console.print("\n[bold]Ingestion complete![/bold]")
    stats = vector_store.get_stats()

    table = Table(title="Ingestion Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Code Files", str(len(code_files)))
    table.add_row("Documentation Files", str(len(doc_files)))
    table.add_row("Code Chunks", str(stats["code_chunks"]))
    table.add_row("Documentation Chunks", str(stats["doc_chunks"]))
    table.add_row("Storage Location", stats["storage_path"])

    console.print(table)


def query_codebase(
    query: str, data_dir: str = ".codex_data", model: str = "gpt-3.5-turbo"
):
    """
    Query the codebase using the AI assistant.

    Args:
        query: Query text
        data_dir: Directory containing processed codebase data
        model: LLM model to use for generating responses
    """
    show_header()
    console.print(f"[bold green]Query[/bold green]: {query}")

    # Set up the query engine
    with Progress(
        SpinnerColumn(), TextColumn("[bold blue]{task.description}"), console=console
    ) as progress:
        task_id = progress.add_task(
            "[bold blue]Initializing query engine...", total=None
        )

        # Set the model name in environment variable
        if model:
            os.environ["OPENAI_API_KEY_MODEL"] = model

        # Initialize the query engine
        engine = QueryEngine(data_dir=data_dir)

        # Update progress
        progress.update(
            task_id, description="[bold blue]Searching for relevant context..."
        )

        # Generate response
        start_time = time.time()
        response = engine.query(query)
        elapsed_time = time.time() - start_time

        progress.update(
            task_id,
            description=f"[bold blue]Generated response in {elapsed_time:.2f} seconds",
        )

    # Print the response
    console.print("\nAnswer:")
    console.print(Panel(Markdown(response), expand=False))


def interactive_chat(data_dir: str = ".codex_data", model: str = "gpt-3.5-turbo", context_window: int = 200000):
    """
    Start an interactive chat session with the AI assistant.
    
    This mode keeps the model loaded between queries for faster response times.

    Args:
        data_dir: Directory containing processed codebase data
        model: LLM model to use for generating responses
        context_window: Size of the context window for the model
    """
    show_header()
    console.print("[bold green]Interactive Chat Mode[/bold green]")
    console.print("Type your questions about the codebase. Type 'exit' or 'quit' to end the session.")
    
    # Set environment variables
    if model:
        os.environ["OPENAI_API_KEY_MODEL"] = model
    
    # Set context window size
    os.environ["CONTEXT_WINDOW_SIZE"] = str(context_window)
    
    # Initialize the query engine once
    with Progress(
        SpinnerColumn(), TextColumn("[bold blue]{task.description}"), console=console
    ) as progress:
        task_id = progress.add_task(
            "[bold blue]Initializing query engine and loading model...", total=None
        )
        
        # Initialize the query engine
        engine = QueryEngine(data_dir=data_dir)
        
        progress.update(task_id, description="[bold blue]Model loaded successfully!")
    
    console.print("[green]Model loaded successfully! You can now ask questions.[/green]")
    
    # Chat history for context
    chat_history = []
    
    # Main chat loop
    while True:
        # Get user input
        console.print("\n[bold cyan]You:[/bold cyan] ", end="")
        query = input()
        
        # Check for exit command
        if query.lower() in ["exit", "quit", "q", "bye"]:
            console.print("[yellow]Exiting chat mode. Goodbye![/yellow]")
            break
        
        # Skip empty queries
        if not query.strip():
            continue
            
        # Add to chat history
        chat_history.append(f"User: {query}")
        
        # Generate response
        console.print("[bold purple]Codex:[/bold purple] ", end="")
        
        start_time = time.time()
        response = engine.query(query)
        elapsed_time = time.time() - start_time
        
        # Print the response
        console.print(Panel(Markdown(response), expand=False))
        console.print(f"[dim](Response generated in {elapsed_time:.2f} seconds)[/dim]")
        
        # Add to chat history
        chat_history.append(f"Assistant: {response}")


def show_stats(data_dir: str = ".codex_data"):
    """
    Show statistics about the processed codebase.

    Args:
        data_dir: Directory containing processed codebase data
    """
    show_header()
    console.print(f"[bold]Loading stats from[/bold]: {data_dir}")

    try:
        # Set up the vector store
        vector_store = VectorStore(data_dir)
        stats = vector_store.get_stats()

        # Create a table with the stats
        table = Table(title="Codex Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Code Files", str(stats.get("code_files", 0)))
        table.add_row("Documentation Files", str(stats.get("doc_files", 0)))
        table.add_row("Code Chunks", str(stats.get("code_chunks", 0)))
        table.add_row("Documentation Chunks", str(stats.get("doc_chunks", 0)))
        table.add_row("Storage Location", stats.get("storage_path", data_dir))
        table.add_row("Last Updated", stats.get("last_updated", "Unknown"))

        console.print(table)
    except Exception as e:
        console.print(f"[bold red]Error loading stats[/bold red]: {str(e)}")


def main():
    """Main entry point for the CLI."""
    setup_logger()

    # Command group
    @click.group()
    def cli():
        """Codex: Make your codebase accessible via AI."""
        pass

    # Ingest command
    @cli.command()
    @click.option(
        "--path",
        "-p",
        required=True,
        type=click.Path(exists=True),
        help="Path to the codebase directory",
    )
    @click.option(
        "--output",
        "-o",
        default=".codex_data",
        help="Directory to store the processed data",
    )
    @click.option(
        "--code-chunk-size",
        type=int,
        default=1000,
        help="Maximum chunk size for code files",
    )
    @click.option(
        "--doc-chunk-size",
        type=int,
        default=1500,
        help="Maximum chunk size for documentation files",
    )
    def ingest(path, output, code_chunk_size, doc_chunk_size):
        """Ingest a codebase for processing."""
        ingest_codebase(path, output, code_chunk_size, doc_chunk_size)

    # Query command
    @cli.command()
    @click.argument("query_text")
    @click.option(
        "--data-dir",
        "-d",
        default=".codex_data",
        help="Directory with processed codebase data",
    )
    @click.option(
        "--model",
        "-m",
        default="gpt-3.5-turbo",
        help="LLM model to use for generating responses",
    )
    def query(query_text, data_dir, model):
        """Query the AI about your codebase."""
        query_codebase(query_text, data_dir, model)
        
    # Interactive chat command
    @cli.command()
    @click.option(
        "--data-dir",
        "-d",
        default=".codex_data",
        help="Directory with processed codebase data",
    )
    @click.option(
        "--model",
        "-m",
        default="gpt-3.5-turbo",
        help="LLM model to use for generating responses",
    )
    @click.option(
        "--context-window",
        "-c",
        default=200000,
        help="Size of the context window for the model",
    )
    def chat(data_dir, model, context_window):
        """Start an interactive chat session with the AI about your codebase."""
        interactive_chat(data_dir, model, context_window)

    # Stats command
    @cli.command()
    @click.option(
        "--data-dir",
        "-d",
        default=".codex_data",
        help="Directory with processed codebase data",
    )
    def stats(data_dir):
        """Show statistics about the processed codebase."""
        show_stats(data_dir)

    # Run the CLI
    cli()


if __name__ == "__main__":
    main()
