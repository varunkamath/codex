#!/usr/bin/env python
"""
Reset ChromaDB script for Codex.

This script completely resets the ChromaDB data and sets up the environment properly.
"""

import os
import sys
import shutil
from pathlib import Path
import subprocess

try:
    from rich.console import Console
    from rich.panel import Panel
except ImportError:
    print("Required packages not found. Please install them with:")
    print("poetry install")
    sys.exit(1)

console = Console()


def reset_chroma():
    """Reset ChromaDB data and set up the environment properly."""
    console.print(Panel("[bold blue]Codex ChromaDB Reset[/bold blue]", expand=False))
    console.print(
        "This script will completely reset the ChromaDB data and set up the environment properly."
    )

    # Check if running on Jetson
    is_jetson = False
    try:
        with open("/proc/device-tree/model", "r") as f:
            model_info = f.read()
            is_jetson = "NVIDIA Jetson" in model_info
            if is_jetson:
                console.print(f"[green]Detected Jetson platform: {model_info.strip()}[/green]")
            else:
                console.print("[yellow]This doesn't appear to be a Jetson device.[/yellow]")
                proceed = console.input("Continue anyway? [y/N]: ").lower() == "y"
                if not proceed:
                    console.print("[yellow]Reset cancelled.[/yellow]")
                    return
    except:
        console.print("[yellow]Could not detect if this is a Jetson device.[/yellow]")
        proceed = console.input("Continue anyway? [y/N]: ").lower() == "y"
        if not proceed:
            console.print("[yellow]Reset cancelled.[/yellow]")
            return

    # Delete ChromaDB data
    chroma_cache = Path(".codex_data/chroma")
    if chroma_cache.exists():
        try:
            shutil.rmtree(chroma_cache)
            console.print("[green]ChromaDB cache deleted successfully.[/green]")
        except Exception as e:
            console.print(f"[red]Error deleting ChromaDB cache: {str(e)}[/red]")
            console.print("[yellow]Please manually delete the .codex_data/chroma directory.[/yellow]")
            return

    # Create .env file with optimized settings
    env_file = Path(".env")
    
    env_content = """# Codex Environment Variables - Jetson Configuration

# Local model configuration
LOCAL_MODEL_PATH=models/llama-2-7b-chat.Q4_K_M.gguf

# Vector store directory
PERSIST_DIRECTORY=.codex_data/chroma

# Avoid tokenizers parallelism warning
TOKENIZERS_PARALLELISM=false

# ONNX Runtime settings for Jetson
OMP_NUM_THREADS=1
OMP_WAIT_POLICY=PASSIVE
OMP_PROC_BIND=FALSE
ONNXRUNTIME_DISABLE_CPU_AFFINITY=1

# Limit memory usage
CHROMADB_TOTAL_MEMORY_LIMIT=4G

# ChromaDB settings
CHROMADB_TELEMETRY_ENABLED=false

# Embedding model settings
EMBEDDING_MODEL=paraphrase-MiniLM-L3-v2
EMBEDDING_BATCH_SIZE=1
"""

    with open(env_file, "w") as f:
        f.write(env_content)

    console.print("[green]Environment file updated with optimized settings.[/green]")

    # Install required packages
    console.print("\n[bold]Installing required packages...[/bold]")
    try:
        subprocess.run(["pip", "install", "sentence-transformers==2.2.2"], check=True)
        console.print("[green]Sentence Transformers installed successfully.[/green]")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not install packages: {str(e)}[/yellow]")

    # Provide next steps
    console.print("\n[bold]Next Steps:[/bold]")
    console.print("1. Run the demo with very small chunk sizes:")
    console.print("   [cyan]poetry run python demo.py --code-chunk-size 300 --doc-chunk-size 500[/cyan]")
    console.print("\n[yellow]Note: Using very small chunk sizes helps prevent memory issues on Jetson devices.[/yellow]")


if __name__ == "__main__":
    reset_chroma() 