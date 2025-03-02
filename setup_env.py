#!/usr/bin/env python
"""
Environment setup script for Codex.
This script sets up the required environment variables for Codex.
"""

import os
import getpass
from pathlib import Path
from rich.console import Console

console = Console()


def setup_environment():
    """Set up environment variables for Codex."""
    console.print("[bold blue]Codex Environment Setup[/bold blue]")
    console.print(
        "This script will help you set up the required environment variables for Codex."
    )

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

    # Get OpenAI API key
    openai_api_key = getpass.getpass("\nEnter your OpenAI API key: ")
    if not openai_api_key:
        console.print("[yellow]No API key provided. Setup cancelled.[/yellow]")
        return

    # Get model name (with default)
    model_name = console.input("\nEnter model name to use [gpt-3.5-turbo]: ")
    model_name = model_name if model_name else "gpt-3.5-turbo"

    # Create .env file
    with open(env_file, "w") as f:
        f.write("# Codex Environment Variables\n")
        f.write(f"OPENAI_API_KEY={openai_api_key}\n")
        f.write(f"OPENAI_MODEL_NAME={model_name}\n")
        f.write("PERSIST_DIRECTORY=.codex_data/chroma\n")
        f.write("# Avoid tokenizers parallelism warning\n")
        f.write("TOKENIZERS_PARALLELISM=false\n")

    console.print("\n[green]Environment setup complete![/green]")
    console.print("The .env file has been created with your settings.")
    console.print(
        "\n[yellow]Note: The .env file contains sensitive information. Do not commit it to version control.[/yellow]"
    )


if __name__ == "__main__":
    setup_environment()
