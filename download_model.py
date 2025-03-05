#!/usr/bin/env python
"""
Model Downloader for Codex.

This script downloads a GGUF model for use with Codex locally.
"""

import sys
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
        "description": "Llama 2 7B Chat (Q4_K_M) - Good balance of quality and speed",
    },
    "llama2-7b-small": {
        "url": "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q2_K.gguf",
        "size": "2.9 GB",
        "filename": "llama-2-7b-chat.Q2_K.gguf",
        "description": "Llama 2 7B Chat (Q2_K) - Smaller and faster, lower quality",
    },
    "tinyllama": {
        "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "size": "0.7 GB",
        "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "description": "TinyLlama 1.1B Chat (Q4_K_M) - Very small and fast, basic capabilities",
    },
    "llama3-8b": {
        "url": "https://huggingface.co/TheBloke/Llama-3-8B-Instruct-GGUF/resolve/main/llama-3-8b-instruct.Q5_K_M.gguf",
        "size": "5.3 GB",
        "filename": "llama-3-8b-instruct.Q5_K_M.gguf",
        "description": "Llama 3 8B Instruct (Q5_K_M) - Latest model from Meta, excellent performance",
    },
    "hermes-3-llama-3.1-8b": {
        "url": "https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B-GGUF/resolve/main/Hermes-3-Llama-3.1-8B.Q6_K.gguf",
        "size": "5.4 GB",
        "filename": "hermes-3-llama-3.1-8b.Q5_K_M.gguf",
        "description": "Hermes 3 (Q5_K_M) - Advanced agentic capabilities, excellent for Jetson GPU",
    },
    "llama3-8b-instruct-coder": {
        "url": "https://huggingface.co/bartowski/Llama-3-8B-Instruct-Coder-v2-GGUF/resolve/main/Llama-3-8B-Instruct-Coder-v2-Q6_K.gguf",
        "size": "5.3 GB",
        "filename": "llama-3-8b-instruct-coder.Q6_K.gguf",
        "description": "Llama 3 8B Instruct Coder (Q6_K) - Latest model + code capabilities, excellent performance with GPU",
    },
    "mistral-7b-v0.2": {
        "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q5_K_M.gguf",
        "size": "5.1 GB",
        "filename": "mistral-7b-instruct-v0.2.Q5_K_M.gguf",
        "description": "Mistral 7B Instruct v0.2 (Q5_K_M) - Excellent instruction following, GPU-friendly",
    },
    "falcon-7b": {
        "url": "https://huggingface.co/TheBloke/falcon-7b-instruct-GGUF/resolve/main/falcon-7b-instruct.Q5_K_M.gguf",
        "size": "4.9 GB",
        "filename": "falcon-7b-instruct.Q5_K_M.gguf",
        "description": "Falcon 7B Instruct (Q5_K_M) - Powerful model from TII, works well on Jetson",
    },
    "codellama-7b": {
        "url": "https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF/resolve/main/codellama-7b-instruct.Q5_K_M.gguf",
        "size": "4.8 GB",
        "filename": "codellama-7b-instruct.Q5_K_M.gguf",
        "description": "CodeLlama 7B Instruct (Q5_K_M) - Specialized for code tasks, GPU-optimized",
    },
    "phi-4": {
        "url": "https://huggingface.co/unsloth/phi-4-GGUF/resolve/main/phi-4-Q8_0.gguf",
        "size": "4.2 GB",
        "filename": "phi-4-Q8_0.gguf",
        "description": "Phi-4 (Q8_0) - Microsoft's latest powerful model with excellent reasoning, ChatML format",
        "prompt_format": "ChatML",
    },
    "mistral-small-24b": {
        "url": "https://huggingface.co/sm54/Mistral-Small-24B-Instruct-2501-Q6_K-GGUF/resolve/main/mistral-small-24b-instruct-2501-q6_k.gguf",
        "size": "14.2 GB",
        "filename": "mistral-small-24b-instruct-2501-q6_k.gguf",
        "description": "Mistral Small 24B Instruct (Q6_K) - Powerful instruction model with V7-Tekken prompt format",
        "prompt_format": "V7-Tekken",
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
            "This script will download a GGUF model for use with Codex locally.",
            title="Welcome",
            border_style="blue",
        )
    )

    # Check if running on Jetson
    is_jetson = False
    jetson_model = ""
    try:
        with open("/proc/device-tree/model", "r") as f:
            model_info = f.read().strip()
            if "NVIDIA Jetson" in model_info:
                is_jetson = True
                jetson_model = model_info
    except:
        pass

    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Group models by category
    standard_models = ["llama2-7b", "llama2-7b-small", "tinyllama"]
    high_performance_models = ["llama3-8b", "mistral-7b-v0.2", "falcon-7b", "codellama-7b", "hermes-3-llama-3.1-8b"]
    large_models = ["llama3-8b-instruct-coder", "phi-4", "mistral-small-24b"]

    # Set default model based on platform
    default_model = "llama3-8b" if is_jetson else "llama2-7b"

    # Display available models
    console.print("\n[bold]Available Models:[/bold]")
    
    if is_jetson:
        console.print(f"\n[green]Detected Jetson platform: {jetson_model}[/green]")
        console.print("[yellow]Recommending GPU-optimized models for your Jetson device[/yellow]")
    
    console.print("\n[bold cyan]Standard Models:[/bold cyan]")
    for key in standard_models:
        model = AVAILABLE_MODELS[key]
        console.print(f"  [green]{key}[/green]: {model['description']} ({model['size']})")
    
    console.print("\n[bold cyan]High Performance Models:[/bold cyan] [yellow](Suitable for Jetson with GPU)[/yellow]")
    for key in high_performance_models:
        model = AVAILABLE_MODELS[key]
        console.print(f"  [green]{key}[/green]: {model['description']} ({model['size']})")
    
    console.print("\n[bold cyan]Large Models:[/bold cyan] [yellow](Require Jetson Orin with good cooling)[/yellow]")
    for key in large_models:
        model = AVAILABLE_MODELS[key]
        console.print(f"  [green]{key}[/green]: {model['description']} ({model['size']})")

    # Get user choice
    console.print(f"\nEnter the model to download (default: {default_model}):")
    choice = input("> ").strip() or default_model

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
        
        # Jetson-specific instructions
        if is_jetson:
            console.print("\n[bold yellow]Jetson-Specific Setup:[/bold yellow]")
            console.print("1. For GPU acceleration, run our optimization script:")
            console.print("   [cyan]poetry run python jetson_gpu_optimize.py[/cyan]")
            console.print("2. Or compile llama-cpp-python with CUDA support manually:")
            console.print("   [cyan]poetry run bash -c 'CUDACXX=/usr/local/cuda/bin/nvcc CMAKE_ARGS=\"-DGGML_CUDA=on\" FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir'[/cyan]")
            console.print("3. Run with Jetson optimizations:")
            console.print("   [cyan]poetry run python demo.py --jetson[/cyan]")
        else:
            console.print("2. Run the demo:")
            console.print("   [cyan]poetry run python demo.py[/cyan]")
        
        console.print("\nThe demo will now use your local model instead of OpenAI.")

    except Exception as e:
        console.print(f"[red]Error downloading model: {str(e)}[/red]")


if __name__ == "__main__":
    main()
