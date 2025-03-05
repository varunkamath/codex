#!/usr/bin/env python
"""
Jetson GPU Check and Optimization Script for Codex.

This script checks GPU availability on Jetson devices and optimizes settings for LLM inference.
"""

import sys
import subprocess
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


def check_jetson_platform():
    """Check if running on a Jetson platform and identify the model."""
    try:
        with open("/proc/device-tree/model", "r") as f:
            model_info = f.read().strip()
            if "NVIDIA Jetson" in model_info:
                return model_info
    except:  # noqa: E722
        pass
    return None


def check_cuda_availability():
    """Check if CUDA is available and get version information."""
    try:
        # Check for nvcc
        nvcc_output = subprocess.check_output(
            ["nvcc", "--version"], stderr=subprocess.STDOUT, universal_newlines=True
        )

        # Extract CUDA version
        for line in nvcc_output.split("\n"):
            if "release" in line and "V" in line:
                return line.strip()
    except:  # noqa: E722
        pass

    # Alternative check for CUDA libraries
    try:
        result = subprocess.check_output(
            ["ldconfig", "-p"], stderr=subprocess.STDOUT, universal_newlines=True
        )
        if "libcuda.so" in result:
            return "CUDA libraries found (version unknown)"
    except:  # noqa: E722
        pass

    return None


def get_gpu_info():
    """Get GPU information using tegrastats."""
    try:
        # Run tegrastats for a brief moment to capture data
        result = subprocess.check_output(
            ["sudo", "tegrastats", "--interval", "1000", "--count", "1"],
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )

        # Parse the output to extract GPU information
        gpu_info = {}
        for line in result.split("\n"):
            if "GR3D_FREQ" in line:
                parts = line.split()
                for part in parts:
                    if "GR3D_FREQ" in part:
                        gpu_info["frequency"] = part.split("=")[1]
            if "RAM" in line:
                parts = line.split()
                for part in parts:
                    if part.startswith("RAM"):
                        ram_parts = part.split("=")[1].split("/")
                        gpu_info["ram_used"] = ram_parts[0]
                        gpu_info["ram_total"] = ram_parts[1]

        return gpu_info
    except Exception as e:
        console.print(f"[yellow]Error getting GPU info: {str(e)}[/yellow]")
        return {}


def optimize_env_settings():
    """Update .env file with optimal GPU settings."""
    env_file = Path(".env")

    # Load existing .env file
    if env_file.exists():
        load_dotenv()

    # Determine optimal settings based on Jetson model
    jetson_model = check_jetson_platform()

    if "Orin" in jetson_model:
        n_gpu_layers = 24
        console.print(
            "[green]Detected Jetson Orin - using optimal settings for this platform[/green]"
        )
    elif "Xavier" in jetson_model:
        n_gpu_layers = 16
        console.print(
            "[green]Detected Jetson Xavier - using optimal settings for this platform[/green]"
        )
    elif "Nano" in jetson_model:
        n_gpu_layers = 8
        console.print(
            "[yellow]Detected Jetson Nano - limited GPU acceleration available[/yellow]"
        )
    else:
        n_gpu_layers = 24  # Default for unknown Jetson models
        console.print("[yellow]Unknown Jetson model - using default settings[/yellow]")

    # Update .env file
    with open(env_file, "a+") as f:
        f.seek(0)
        content = f.read()

        # Add or update GPU settings
        if "N_GPU_LAYERS" not in content:
            f.write("\n# GPU acceleration settings for LLM\n")
            f.write(f"N_GPU_LAYERS={n_gpu_layers}\n")
            f.write(f"GPU_LAYERS_DRAFT={n_gpu_layers}\n")
            console.print("[green]Added GPU acceleration settings to .env file[/green]")
        else:
            console.print("[yellow]GPU settings already exist in .env file[/yellow]")

    return n_gpu_layers


def run_jetson_clocks():
    """Run jetson_clocks to maximize performance."""
    try:
        console.print("[bold]Running jetson_clocks to maximize performance...[/bold]")
        _ = subprocess.check_output(
            ["sudo", "jetson_clocks"], stderr=subprocess.STDOUT, universal_newlines=True
        )
        console.print("[green]Successfully maximized Jetson performance[/green]")
        return True
    except Exception as e:
        console.print(f"[yellow]Error running jetson_clocks: {str(e)}[/yellow]")
        console.print(
            "[yellow]You may need to run 'sudo jetson_clocks' manually to maximize performance[/yellow]"
        )
        return False


def main():
    """Main function to check GPU and optimize settings."""
    console.print(
        Panel(
            "[bold blue]Jetson GPU Check and Optimization[/bold blue]",
            subtitle="For Codex LLM Acceleration",
        )
    )

    # Check if running on Jetson
    jetson_model = check_jetson_platform()
    if not jetson_model:
        console.print(
            "[red]This script is intended for NVIDIA Jetson platforms only.[/red]"
        )
        sys.exit(1)

    console.print(f"[green]Detected Jetson platform: {jetson_model}[/green]")

    # Check CUDA availability
    cuda_version = check_cuda_availability()
    if cuda_version:
        console.print(f"[green]CUDA is available: {cuda_version}[/green]")
    else:
        console.print("[red]CUDA not detected. GPU acceleration may not work.[/red]")
        console.print(
            "[yellow]Please ensure CUDA is properly installed on your Jetson.[/yellow]"
        )

    # Get GPU information
    gpu_info = get_gpu_info()
    if gpu_info:
        table = Table(title="GPU Information")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        for key, value in gpu_info.items():
            table.add_row(key, value)

        console.print(table)

    # Optimize environment settings
    _ = optimize_env_settings()

    # Ask to run jetson_clocks
    if (
        console.input("\nRun jetson_clocks to maximize performance? [y/N]: ").lower()
        == "y"
    ):
        run_jetson_clocks()

    # Provide next steps
    console.print("\n[bold]Next Steps:[/bold]")
    console.print("1. Run the demo with GPU acceleration:")
    console.print(
        "   [cyan]poetry run python demo.py --code-chunk-size 500 --doc-chunk-size 800[/cyan]"
    )

    console.print("\n2. Verify GPU is being used by checking for lines like:")
    console.print("   [cyan]load_tensors: layer X assigned to device CUDA[/cyan]")
    console.print("   in the startup logs.")

    console.print(
        "\n[yellow]Note: If you don't see GPU being used, try restarting your Jetson device.[/yellow]"
    )


if __name__ == "__main__":
    main()
