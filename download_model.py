#!/usr/bin/env python
"""
Model Downloader for Codex.

This script downloads a GGUF model for use with Codex locally.
"""

import os
import sys
import argparse
import requests
from pathlib import Path
from tqdm import tqdm

try:
    from rich.console import Console
    from rich.panel import Panel
except ImportError:
    print("Required packages not found. Please install them with:")
    print("poetry install")
    sys.exit(1)

console = Console()

# Available models with their URLs and file sizes
AVAILABLE_MODELS = {
    "llama2-7b": {
        "url": "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf",
        "size": "4.1 GB",
        "filename": "llama-2-7b-chat.Q4_K_M.gguf",
        "description": "Llama 2 7B Chat - Good balance of speed and quality",
    },
    "llama2-7b-small": {
        "url": "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q2_K.gguf",
        "size": "2.9 GB",
        "filename": "llama-2-7b-chat.Q2_K.gguf",
        "description": "Llama 2 7B Chat (Smaller version) - Faster but lower quality",
    },
    "tinyllama": {
        "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "size": "0.7 GB",
        "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "description": "TinyLlama 1.1B - Very small and fast, but lower quality responses",
    },
}


def download_file(url, filename):
    """Download a file from a URL with a progress bar."""
    console.print(f"[blue]Downloading {filename}...[/blue]")

    # Create the models directory if it doesn't exist
    Path("models").mkdir(exist_ok=True)

    filepath = Path("models") / filename

    # Check if file already exists
    if filepath.exists():
        overwrite = (
            console.input(
                f"[yellow]{filename} already exists. Overwrite? [y/N]: [/yellow]"
            ).lower()
            == "y"
        )
        if not overwrite:
            console.print("[green]Using existing file.[/green]")
            return filepath

    # Stream download with progress bar
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte

    with (
        open(filepath, "wb") as f,
        tqdm(total=total_size, unit="iB", unit_scale=True, desc=filename) as bar,
    ):
        for data in response.iter_content(block_size):
            bar.update(len(data))
            f.write(data)

    console.print(f"[green]Download complete: {filepath}[/green]")
    return filepath


def update_env_file(model_path):
    """Update the .env file with the new model path."""
    env_file = Path(".env")

    # Check if .env exists
    if not env_file.exists():
        console.print("[yellow].env file not found. Creating a new one.[/yellow]")
        with open(env_file, "w") as f:
            f.write("# Codex Environment Variables\n")
            f.write(f"LOCAL_MODEL_PATH={model_path}\n")
            f.write("TOKENIZERS_PARALLELISM=false\n")
        console.print("[green].env file created.[/green]")
        return

    # Read the existing .env file
    with open(env_file, "r") as f:
        lines = f.readlines()

    # Check if LOCAL_MODEL_PATH already exists
    local_model_path_exists = False
    for i, line in enumerate(lines):
        if line.strip().startswith("LOCAL_MODEL_PATH="):
            lines[i] = f"LOCAL_MODEL_PATH={model_path}\n"
            local_model_path_exists = True
            break

    # If LOCAL_MODEL_PATH doesn't exist, add it
    if not local_model_path_exists:
        lines.append(f"LOCAL_MODEL_PATH={model_path}\n")

    # Write the updated .env file
    with open(env_file, "w") as f:
        f.writelines(lines)

    console.print("[green].env file updated with the new model path.[/green]")


def main():
    """Main function."""
    console.print(
        Panel.fit(
            "[bold blue]Codex Model Downloader[/bold blue]\n\n"
            "This tool downloads a model for local use with Codex.",
            title="Welcome",
            border_style="blue",
        )
    )

    # List available models
    console.print("\n[bold]Available Models:[/bold]")
    for key, model in AVAILABLE_MODELS.items():
        console.print(
            f"  [cyan]{key}[/cyan] - {model['description']} ({model['size']})"
        )

    # Get user choice
    choice = console.input("\n[bold]Choose a model to download [llama2-7b]: [/bold]")
    choice = choice.strip().lower() if choice.strip() else "llama2-7b"

    if choice not in AVAILABLE_MODELS:
        console.print(f"[red]Invalid choice: {choice}[/red]")
        console.print(
            "[yellow]Please run the script again and select a valid model.[/yellow]"
        )
        return

    # Download the model
    model_info = AVAILABLE_MODELS[choice]
    try:
        filepath = download_file(model_info["url"], model_info["filename"])

        # Update the .env file
        update_env_file(str(filepath))

        # Show instructions
        console.print("\n[bold green]Model downloaded successfully![/bold green]")
        console.print("\n[bold]Next Steps:[/bold]")
        console.print("1. Make sure you have the required packages:")
        console.print("   [cyan]poetry install[/cyan]")
        console.print("   [cyan]poetry add langchain-community llama-cpp-python[/cyan]")
        console.print("2. Run the demo:")
        console.print("   [cyan]poetry run python demo.py[/cyan]")
        console.print("\nThe demo will now use your local model instead of OpenAI.")

    except Exception as e:
        console.print(f"[red]Error downloading model: {str(e)}[/red]")


if __name__ == "__main__":
    main()
