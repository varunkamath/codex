#!/usr/bin/env python
"""
Jetson-specific setup script for Codex.

This script configures Codex for optimal performance on NVIDIA Jetson platforms.
"""

import os
import sys
import shutil
from pathlib import Path

try:
    from rich.console import Console
    from rich.panel import Panel
except ImportError:
    print("Required packages not found. Please install them with:")
    print("poetry install")
    sys.exit(1)

console = Console()


def setup_jetson_environment():
    """Set up environment variables and configuration for Jetson."""
    console.print("[bold blue]Codex Jetson Setup[/bold blue]")
    console.print(
        "This script will configure Codex for optimal performance on NVIDIA Jetson platforms."
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
                    console.print("[yellow]Setup cancelled.[/yellow]")
                    return
    except:
        console.print("[yellow]Could not detect if this is a Jetson device.[/yellow]")
        proceed = console.input("Continue anyway? [y/N]: ").lower() == "y"
        if not proceed:
            console.print("[yellow]Setup cancelled.[/yellow]")
            return

    # Check if .env file exists
    env_file = Path(".env")
    if env_file.exists():
        overwrite = (
            console.input("\n.env file already exists. Overwrite? [y/N]: ").lower()
            == "y"
        )
        if not overwrite:
            console.print("[yellow]Setup cancelled. Using existing .env file.[/yellow]")
            return

    # Create .env file with Jetson-optimized settings
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

# GPU acceleration settings for LLM
N_GPU_LAYERS=24
GPU_LAYERS_DRAFT=24
"""

    with open(env_file, "w") as f:
        f.write(env_content)

    console.print("\n[green]Environment setup complete![/green]")
    console.print("The .env file has been created with Jetson-optimized settings.")

    # Handle ChromaDB migration
    chroma_cache = Path(".codex_data/chroma")
    if chroma_cache.exists():
        console.print("\n[bold red]WARNING: ChromaDB Migration Required[/bold red]")
        console.print("ChromaDB has changed its architecture and requires migration.")
        console.print("The safest approach is to clear the existing data and start fresh.")
        
        clear_cache = console.input("\nClear existing ChromaDB data? [Y/n]: ").lower() != "n"
        
        if clear_cache:
            try:
                shutil.rmtree(chroma_cache)
                console.print("[green]ChromaDB cache cleared successfully.[/green]")
            except Exception as e:
                console.print(f"[red]Error clearing cache: {str(e)}[/red]")
                console.print("[yellow]Please manually delete the .codex_data/chroma directory.[/yellow]")
        else:
            console.print("[yellow]Keeping existing data. This may cause errors.[/yellow]")
            console.print("[yellow]If you encounter errors, run this command:[/yellow]")
            console.print("[cyan]rm -rf .codex_data/chroma[/cyan]")
    
    # Provide next steps
    console.print("\n[bold]Next Steps:[/bold]")
    console.print("1. Run the demo with reduced chunk sizes:")
    console.print("   [cyan]poetry run python demo.py --code-chunk-size 500 --doc-chunk-size 800[/cyan]")
    console.print("\n2. Or ingest your own codebase with:")
    console.print("   [cyan]poetry run python -m codex.main ingest --path /path/to/codebase --code-chunk-size 500 --doc-chunk-size 800[/cyan]")
    console.print("\n[yellow]Note: Smaller chunk sizes help prevent memory issues on Jetson devices.[/yellow]")


if __name__ == "__main__":
    setup_jetson_environment() 