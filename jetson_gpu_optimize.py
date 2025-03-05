#!/usr/bin/env python
"""
Jetson GPU Optimization Script for Codex.

This script optimizes Codex for GPU acceleration on NVIDIA Jetson platforms.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from dotenv import load_dotenv, set_key

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
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
    except:
        pass
    return None

def check_cuda_availability():
    """Check if CUDA is available and get version information."""
    try:
        # Check for nvcc
        nvcc_output = subprocess.check_output(["nvcc", "--version"], 
                                             stderr=subprocess.STDOUT, 
                                             universal_newlines=True)
        
        # Extract CUDA version
        for line in nvcc_output.split('\n'):
            if "release" in line and "V" in line:
                return line.strip()
    except:
        pass
    
    # Alternative check for CUDA libraries
    try:
        result = subprocess.check_output(["ldconfig", "-p"], 
                                        stderr=subprocess.STDOUT, 
                                        universal_newlines=True)
        if "libcuda.so" in result:
            return "CUDA libraries found (version unknown)"
    except:
        pass
    
    return None

def get_gpu_info():
    """Get GPU information using tegrastats."""
    try:
        # Run tegrastats for a brief moment to capture data
        result = subprocess.check_output(["sudo", "tegrastats", "--interval", "1000", "--count", "1"], 
                                        stderr=subprocess.STDOUT, 
                                        universal_newlines=True)
        
        # Parse the output to extract GPU information
        gpu_info = {}
        for line in result.split('\n'):
            if "GR3D_FREQ" in line:
                parts = line.split()
                for part in parts:
                    if "GR3D_FREQ" in part:
                        gpu_info["frequency"] = part.split('=')[1]
            if "RAM" in line:
                parts = line.split()
                for part in parts:
                    if part.startswith("RAM"):
                        ram_parts = part.split('=')[1].split('/')
                        gpu_info["ram_used"] = ram_parts[0]
                        gpu_info["ram_total"] = ram_parts[1]
        
        return gpu_info
    except Exception as e:
        console.print(f"[yellow]Error getting GPU info: {str(e)}[/yellow]")
        return {}

def check_model_compatibility():
    """Check if the current model is compatible with GPU acceleration on Jetson."""
    # Load environment variables
    load_dotenv()
    
    # Get current model path
    model_path = os.environ.get("LOCAL_MODEL_PATH", "")
    if not model_path:
        console.print("[yellow]No model path found in environment variables.[/yellow]")
        return None, []
    
    model_file = Path(model_path)
    if not model_file.exists():
        console.print(f"[yellow]Model file not found at {model_path}[/yellow]")
        return None, []
    
    # Check model filename for compatibility issues
    model_name = model_file.name.lower()
    
    # Models known to have issues with Jetson GPU acceleration
    problematic_models = [
        ("mixtral", "Mixtral models use a Mixture of Experts architecture that may not be fully compatible with llama-cpp-python on Jetson"),
        ("mpt", "MPT models have a different architecture that may cause issues with CUDA acceleration on Jetson"),
        ("q2_k", "Q2_K quantization is too aggressive and may not work well with GPU acceleration"),
        ("q3_k", "Q3_K quantization may have issues with GPU acceleration on Jetson")
    ]
    
    # Models known to work well with Jetson GPU acceleration
    recommended_models = [
        "llama-3-8b-instruct.q5_k_m.gguf",
        "llama-3-8b-instruct.q6_k.gguf",
        "mistral-7b-instruct-v0.2.q5_k_m.gguf",
        "falcon-7b-instruct.q5_k_m.gguf",
        "codellama-7b-instruct.q5_k_m.gguf",
        "phi-2.q5_k_m.gguf",
        "hermes-3-llama-3.1-8b.q5_k_m.gguf"
    ]
    
    # Check for problematic models
    issues = []
    for keyword, reason in problematic_models:
        if keyword in model_name:
            issues.append(f"[yellow]Warning:[/yellow] {reason}")
    
    # Return compatibility status
    if issues:
        return model_name, issues
    else:
        return model_name, []

def compile_llama_cpp_with_cuda():
    """Compile llama-cpp-python with CUDA support."""
    console.print("[bold]Compiling llama-cpp-python with CUDA support...[/bold]")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("[cyan]Uninstalling existing llama-cpp-python...", total=None)
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "llama-cpp-python"])
            
            progress.update(task, description="[cyan]Installing with CUDA support (this may take a while)...")
            env = os.environ.copy()
            env["CUDACXX"] = "/usr/local/cuda/bin/nvcc"
            env["CMAKE_ARGS"] = "-DGGML_CUDA=on"
            env["FORCE_CMAKE"] = "1"
            
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "llama-cpp-python", 
                 "--force-reinstall", "--upgrade", "--no-cache-dir"],
                env=env
            )
            
        console.print("[green]Successfully compiled llama-cpp-python with CUDA support![/green]")
        return True
    except Exception as e:
        console.print(f"[red]Error compiling llama-cpp-python: {str(e)}[/red]")
        console.print("\n[yellow]Try running this command manually:[/yellow]")
        console.print("[cyan]CUDACXX=/usr/local/cuda/bin/nvcc CMAKE_ARGS=\"-DGGML_CUDA=on\" FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir[/cyan]")
        return False

def update_env_file():
    """Update .env file with GPU acceleration settings."""
    console.print("[bold]Updating .env file with GPU acceleration settings...[/bold]")
    
    env_file = Path(".env")
    
    # Load existing .env file
    if env_file.exists():
        load_dotenv(env_file)
    
    # Determine optimal settings based on Jetson model
    jetson_model = check_jetson_platform()
    
    if "Orin" in jetson_model:
        n_gpu_layers = 24
        console.print("[green]Detected Jetson Orin - using optimal settings for this platform[/green]")
    elif "Xavier" in jetson_model:
        n_gpu_layers = 16
        console.print("[green]Detected Jetson Xavier - using optimal settings for this platform[/green]")
    elif "Nano" in jetson_model:
        n_gpu_layers = 8
        console.print("[yellow]Detected Jetson Nano - limited GPU acceleration available[/yellow]")
    else:
        n_gpu_layers = 24  # Default for unknown Jetson models
        console.print("[yellow]Unknown Jetson model - using default settings[/yellow]")
    
    # Update .env file
    with open(env_file, "a+") as f:
        f.seek(0)
        content = f.read()
        
        # Add or update GPU settings
        gpu_settings = [
            "\n# GPU acceleration settings for LLM",
            f"N_GPU_LAYERS={n_gpu_layers}",
            f"GPU_LAYERS_DRAFT={n_gpu_layers}",
            "N_BATCH=512"  # Larger batch size for GPU
        ]
        
        # Check if settings already exist
        if "N_GPU_LAYERS" not in content:
            f.write("\n".join(gpu_settings))
            console.print("[green]Added GPU acceleration settings to .env file[/green]")
        else:
            console.print("[yellow]GPU settings already exist in .env file[/yellow]")
    
    return n_gpu_layers

def run_jetson_clocks():
    """Run jetson_clocks to maximize performance."""
    try:
        console.print("[bold]Running jetson_clocks to maximize performance...[/bold]")
        result = subprocess.check_output(["sudo", "jetson_clocks"], 
                                        stderr=subprocess.STDOUT, 
                                        universal_newlines=True)
        console.print("[green]Successfully maximized Jetson performance[/green]")
        return True
    except Exception as e:
        console.print(f"[yellow]Error running jetson_clocks: {str(e)}[/yellow]")
        console.print("[yellow]You may need to run 'sudo jetson_clocks' manually to maximize performance[/yellow]")
        return False

def main():
    """Main function to optimize Jetson for GPU acceleration."""
    console.print(Panel.fit(
        "[bold blue]Jetson GPU Optimization for Codex[/bold blue]\n\n"
        "This script will optimize Codex for GPU acceleration on your Jetson device.",
        title="Welcome",
        border_style="blue",
    ))
    
    # Check if running on Jetson
    jetson_model = check_jetson_platform()
    if not jetson_model:
        console.print("[red]This script is intended for NVIDIA Jetson platforms only.[/red]")
        sys.exit(1)
    
    console.print(f"[green]Detected Jetson platform: {jetson_model}[/green]")
    
    # Check CUDA availability
    cuda_version = check_cuda_availability()
    if cuda_version:
        console.print(f"[green]CUDA is available: {cuda_version}[/green]")
    else:
        console.print("[red]CUDA not detected. GPU acceleration may not work.[/red]")
        console.print("[yellow]Please ensure CUDA is properly installed on your Jetson.[/yellow]")
        if not console.input("\nContinue anyway? [y/N]: ").lower().startswith('y'):
            sys.exit(1)
    
    # Check model compatibility
    model_name, issues = check_model_compatibility()
    if model_name:
        console.print(f"\n[bold]Current model:[/bold] {model_name}")
        
        if issues:
            console.print("\n[bold yellow]Model Compatibility Issues:[/bold yellow]")
            for issue in issues:
                console.print(f"- {issue}")
            
            console.print("\n[bold green]Recommended Models for Jetson GPU Acceleration:[/bold green]")
            console.print("- Llama 3 8B Instruct (Q5_K_M) - Excellent performance with GPU")
            console.print("- Mistral 7B Instruct v0.2 (Q5_K_M) - Great instruction following")
            console.print("- Falcon 7B Instruct (Q5_K_M) - Good performance on Jetson")
            console.print("- CodeLlama 7B Instruct (Q5_K_M) - Specialized for code tasks")
            console.print("- Hermes 3 Llama 3.1 8B (Q5_K_M) - Advanced agentic capabilities")
            
            console.print("\nYou can download a compatible model with:")
            console.print("[cyan]poetry run python download_model.py[/cyan]")
            
            if not console.input("\nContinue with current model anyway? [y/N]: ").lower().startswith('y'):
                console.print("\nPlease download a compatible model and run this script again.")
                sys.exit(0)
    
    # Get GPU information
    gpu_info = get_gpu_info()
    if gpu_info:
        table = Table(title="GPU Information")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in gpu_info.items():
            table.add_row(key, value)
        
        console.print(table)
    
    # Optimization steps
    steps = [
        ("Compile llama-cpp-python with CUDA support", compile_llama_cpp_with_cuda),
        ("Update .env file with GPU acceleration settings", update_env_file),
    ]
    
    for step_name, step_func in steps:
        console.print(f"\n[bold]Step: {step_name}[/bold]")
        success = step_func()
        if not success and step_name == "Compile llama-cpp-python with CUDA support":
            if not console.input("\nContinue with other optimizations? [y/N]: ").lower().startswith('y'):
                sys.exit(1)
    
    # Ask to run jetson_clocks
    if console.input("\nRun jetson_clocks to maximize performance? [y/N]: ").lower().startswith('y'):
        run_jetson_clocks()
    
    # Provide next steps
    console.print("\n[bold]Optimization Complete![/bold]")
    console.print("\n[bold]Next Steps:[/bold]")
    console.print("1. Run the demo with GPU acceleration:")
    console.print("   [cyan]poetry run python demo.py --jetson[/cyan]")
    
    console.print("\n2. Verify GPU is being used by checking for lines like:")
    console.print("   [cyan]load_tensors: layer X assigned to device CUDA[/cyan]")
    console.print("   in the startup logs.")
    
    console.print("\n3. If you don't see GPU being used, try downloading a different model format:")
    console.print("   [cyan]poetry run python download_model.py[/cyan]")
    console.print("   and select a model with Q5_K_M or Q6_K quantization.")
    
    console.print("\n[yellow]Note: If you still don't see GPU being used, try restarting your Jetson device.[/yellow]")

if __name__ == "__main__":
    main() 