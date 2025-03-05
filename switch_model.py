#!/usr/bin/env python
"""
Model Switcher for Codex.

This script helps switch between different local models for Codex.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
except ImportError:
    print("Required packages not found. Please install them with:")
    print("poetry add rich python-dotenv")
    sys.exit(1)

console = Console()


def find_models():
    """Find all GGUF models in the models directory."""
    models_dir = Path("models")
    if not models_dir.exists():
        return []

    # Find all GGUF files
    model_files = list(models_dir.glob("*.gguf"))

    # Sort by modification time (newest first)
    model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    return model_files


def get_current_model():
    """Get the currently configured model from .env file."""
    load_dotenv()
    model_path = os.environ.get("LOCAL_MODEL_PATH", "")
    if not model_path:
        return None

    return Path(model_path)


def update_env_file(model_path):
    """Update the .env file with the new model path."""
    env_file = Path(".env")

    # Convert Path object to string if needed
    if isinstance(model_path, Path):
        model_path_str = str(model_path)
    else:
        model_path_str = model_path

    # Create .env file if it doesn't exist
    if not env_file.exists():
        env_file.touch()

    # Update the model path
    with open(env_file, "r+") as f:
        content = f.read()

        # Check if LOCAL_MODEL_PATH already exists
        if "LOCAL_MODEL_PATH=" in content:
            # Replace the existing path
            lines = content.split("\n")
            new_lines = []
            for line in lines:
                if line.startswith("LOCAL_MODEL_PATH="):
                    new_lines.append(f"LOCAL_MODEL_PATH={model_path_str}")
                else:
                    new_lines.append(line)

            # Write the updated content
            f.seek(0)
            f.truncate()
            f.write("\n".join(new_lines))
        else:
            # Append the new path
            f.seek(0, os.SEEK_END)
            f.write(f"\nLOCAL_MODEL_PATH={model_path_str}")

    console.print(f"[green]Updated .env file with model path: {model_path_str}[/green]")
    return True


def get_model_info(model_path):
    """Get information about a model from its filename."""
    filename = model_path.name.lower()

    # Determine model family
    model_family = "Unknown"
    if "hermes" in filename:
        model_family = "Hermes"
    elif "llama-3" in filename or "llama3" in filename:
        model_family = "Llama 3"
    elif "llama-2" in filename or "llama2" in filename:
        model_family = "Llama 2"
    elif "mistral" in filename:
        model_family = "Mistral"
    elif "falcon" in filename:
        model_family = "Falcon"
    elif "codellama" in filename:
        model_family = "CodeLlama"
    elif "phi-2" in filename:
        model_family = "Phi-2"
    elif "tinyllama" in filename:
        model_family = "TinyLlama"
    elif "mixtral" in filename:
        model_family = "Mixtral"

    # Determine quantization
    quant = "Unknown"
    if "q2_k" in filename:
        quant = "Q2_K (Very Low)"
    elif "q3_k" in filename:
        quant = "Q3_K (Low)"
    elif "q4_k_m" in filename:
        quant = "Q4_K_M (Medium)"
    elif "q5_k_m" in filename:
        quant = "Q5_K_M (High)"
    elif "q6_k" in filename:
        quant = "Q6_K (Very High)"
    elif "q8_0" in filename:
        quant = "Q8_0 (Highest)"

    # Determine size
    size_mb = model_path.stat().st_size / (1024 * 1024)
    size_gb = size_mb / 1024

    # Determine GPU compatibility
    gpu_compat = (
        "Good"
        if any(q in filename for q in ["q5_k_m", "q6_k"])
        else "Medium"
        if "q4_k_m" in filename
        else "Poor"
        if any(q in filename for q in ["q2_k", "q3_k"])
        else "Unknown"
    )

    # Determine prompt format
    prompt_format = "Standard"
    if "hermes" in filename:
        prompt_format = "ChatML"

    return {
        "family": model_family,
        "quantization": quant,
        "size": f"{size_gb:.1f} GB",
        "gpu_compatibility": gpu_compat,
        "prompt_format": prompt_format,
    }


def main():
    """Main function."""
    console.print(
        Panel.fit(
            "[bold blue]Codex Model Switcher[/bold blue]\n\n"
            "This script helps you switch between different local models for Codex.",
            title="Welcome",
            border_style="blue",
        )
    )

    # Find all models
    model_files = find_models()
    if not model_files:
        console.print("[yellow]No models found in the 'models' directory.[/yellow]")
        console.print("Please download a model first using:")
        console.print("[cyan]poetry run python download_model.py[/cyan]")
        return

    # Get current model
    current_model = get_current_model()

    # Display available models
    console.print("\n[bold]Available Models:[/bold]")

    table = Table(title="Local Models")
    table.add_column("#", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Family", style="blue")
    table.add_column("Quantization", style="magenta")
    table.add_column("Size", style="yellow")
    table.add_column("GPU Compatibility", style="red")
    table.add_column("Prompt Format", style="blue")
    table.add_column("Current", style="bold green")

    for i, model_file in enumerate(model_files, 1):
        model_info = get_model_info(model_file)
        is_current = current_model and model_file.resolve() == current_model.resolve()

        table.add_row(
            str(i),
            model_file.name,
            model_info["family"],
            model_info["quantization"],
            model_info["size"],
            model_info["gpu_compatibility"],
            model_info["prompt_format"],
            "✓" if is_current else "",
        )

    console.print(table)

    # Get user choice
    console.print("\nEnter the number of the model to switch to (or 'q' to quit):")
    choice = input("> ").strip().lower()

    if choice == "q":
        return

    try:
        choice_idx = int(choice) - 1
        if choice_idx < 0 or choice_idx >= len(model_files):
            console.print("[red]Invalid choice. Please enter a valid number.[/red]")
            return

        selected_model = model_files[choice_idx]

        # Check if already using this model
        if current_model and selected_model.resolve() == current_model.resolve():
            console.print(
                f"[yellow]Already using model: {selected_model.name}[/yellow]"
            )
            return

        # Update .env file
        if update_env_file(selected_model):
            console.print(
                f"[green]Successfully switched to model:[/green] {selected_model.name}"
            )

            # Check if it's a Hermes model and provide additional information
            if "hermes" in selected_model.name.lower():
                console.print(
                    "\n[bold yellow]Note:[/bold yellow] You've selected a Hermes model which uses the ChatML prompt format."
                )
                console.print(
                    "The system has been configured to automatically use the correct format."
                )
                console.print("Hermes models have enhanced capabilities for:")
                console.print("- Advanced agentic reasoning")
                console.print("- Better instruction following")
                console.print("- Improved code generation")
                console.print("- Function calling and structured outputs")

            console.print("\n[bold]Next steps:[/bold]")
            console.print("1. Run the demo: [cyan]poetry run python demo.py[/cyan]")
            console.print(
                "2. For Jetson devices: [cyan]poetry run python demo.py --jetson[/cyan]"
            )
        else:
            console.print("[red]Failed to update environment file.[/red]")

        # Check for Jetson platform
        is_jetson = False
        try:
            with open("/proc/device-tree/model", "r") as f:
                model_info = f.read().strip()
                is_jetson = "NVIDIA Jetson" in model_info
        except:  # noqa: E722
            pass

        # Provide next steps
        console.print("\n[bold]Next Steps:[/bold]")

        # Check model compatibility with GPU on Jetson
        model_info = get_model_info(selected_model)
        if is_jetson and model_info["gpu_compatibility"] != "Good":
            console.print(
                "[yellow]Note: This model may not work optimally with GPU acceleration on Jetson.[/yellow]"
            )
            console.print(
                "Consider using a model with Q5_K_M or Q6_K quantization for better GPU performance."
            )

        console.print("1. Run the demo:")
        if is_jetson:
            console.print("   [cyan]poetry run python demo.py --jetson[/cyan]")
        else:
            console.print("   [cyan]poetry run python demo.py[/cyan]")

        if is_jetson:
            console.print("\n2. For GPU acceleration, run:")
            console.print("   [cyan]poetry run python jetson_gpu_optimize.py[/cyan]")

    except ValueError:
        console.print("[red]Invalid input. Please enter a number.[/red]")


if __name__ == "__main__":
    main()
