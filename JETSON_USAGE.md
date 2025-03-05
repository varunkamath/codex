# Using Codex on NVIDIA Jetson Platforms

This guide provides specific instructions for running Codex on NVIDIA Jetson devices (Nano, Xavier, Orin, etc.).

## Quick Start

The easiest way to get started on a Jetson device is to use our Jetson-specific setup and optimization scripts:

```bash
# Run the Jetson setup script for basic configuration
poetry run python jetson_setup.py

# Run the GPU optimization script for CUDA acceleration
poetry run python jetson_gpu_optimize.py

# Run the demo with Jetson-optimized settings
poetry run python demo.py --jetson
```

## GPU Acceleration

Codex now includes full GPU acceleration support for Jetson devices, which can dramatically improve inference speed:

### Automatic GPU Setup

The `jetson_gpu_optimize.py` script automates the process of setting up GPU acceleration:

1. Detects your Jetson model (Orin, Xavier, Nano)
2. Compiles `llama-cpp-python` with CUDA support
3. Configures optimal GPU settings based on your hardware
4. Updates your `.env` file with the right parameters
5. Maximizes performance with `jetson_clocks`

```bash
# Run the GPU optimization script
poetry run python jetson_gpu_optimize.py
```

### Manual GPU Setup

If you prefer to set up GPU acceleration manually:

1. Compile `llama-cpp-python` with CUDA support:
   ```bash
   poetry run bash -c 'CUDACXX=/usr/local/cuda/bin/nvcc CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir'
   ```

2. Add these settings to your `.env` file:
   ```
   # GPU acceleration settings for LLM
   N_GPU_LAYERS=24  # Use 16 for Xavier, 8 for Nano
   GPU_LAYERS_DRAFT=24
   N_BATCH=512
   ```

3. Maximize GPU performance:
   ```bash
   sudo jetson_clocks
   ```

### Verifying GPU Usage

You can verify that GPU acceleration is working when:

1. You see log messages like:
   ```
   load_tensors: layer X assigned to device CUDA
   ```
   during model initialization.

2. Inference speed is significantly faster than CPU-only mode.

3. GPU memory is being utilized (check with `sudo tegrastats`).

### Command Line Options

The demo script now supports GPU-specific command line options:

```bash
# Set specific number of GPU layers
poetry run python demo.py --jetson --gpu-layers 24

# Set batch size for inference
poetry run python demo.py --jetson --batch-size 512
```

## Common Issues on Jetson

Jetson devices have specific hardware characteristics that can cause issues with the default Codex configuration:

1. **CPU Affinity Errors**: The error `pthread_setaffinity_np failed` occurs because ONNX Runtime tries to bind to specific CPU cores that don't exist on Jetson.

2. **Vector Index Out of Bounds**: The error `std::vector<_Tp, _Alloc>::reference std::vector<_Tp, _Alloc>::operator[](size_type) [with _Tp = unsigned int...]: Assertion '__n < this->size()' failed` is related to memory management issues.

3. **Memory Limitations**: Jetson devices have limited RAM compared to desktop systems, which can cause out-of-memory errors during embedding generation.

4. **ChromaDB Migration Error**: The error `You are using a deprecated configuration of Chroma` occurs because ChromaDB has updated its architecture. This requires either clearing your existing data or using the migration tool.

## Optimizations Applied

Our Jetson-specific setup makes the following adjustments:

1. **Disabled CPU Affinity**: Environment variables prevent ONNX Runtime from trying to bind to specific cores.

2. **Smaller Batch Sizes**: Processing data in smaller batches to prevent memory issues.

3. **Reduced Chunk Sizes**: Using smaller text chunks for code and documentation.

4. **Optimized LLM Settings**: Configured LlamaCpp with Jetson-friendly parameters.

5. **Memory-Efficient Embedding**: Using quantized embeddings and processing in smaller batches.

6. **GPU Acceleration**: Automatic detection and configuration of GPU layers for optimal performance.

7. **Dynamic Resource Allocation**: Adjusts settings based on the specific Jetson model detected.

## Manual Configuration

If you need to manually configure Codex for Jetson, add these environment variables to your `.env` file:

```
# ONNX Runtime settings for Jetson
OMP_NUM_THREADS=1
OMP_WAIT_POLICY=PASSIVE
OMP_PROC_BIND=FALSE
ONNXRUNTIME_DISABLE_CPU_AFFINITY=1

# Limit memory usage
CHROMADB_TOTAL_MEMORY_LIMIT=4G

# GPU acceleration settings
N_GPU_LAYERS=24  # Adjust based on your Jetson model
GPU_LAYERS_DRAFT=24
N_BATCH=512
```

## Performance Tips

1. **Use Smaller Chunk Sizes**: When ingesting code, use smaller chunk sizes:
   ```bash
   poetry run python -m codex.main ingest --path /path/to/codebase --code-chunk-size 500 --doc-chunk-size 800
   ```

2. **Clear ChromaDB Cache**: If you encounter issues, try clearing the ChromaDB cache:
   ```bash
   rm -rf .codex_data/chroma
   ```

3. **GPU Acceleration**: For Jetson devices, enable GPU acceleration by setting the following in your `.env` file:
   ```
   # GPU acceleration settings for LLM
   N_GPU_LAYERS=24  # Use 24 for Orin, 16 for Xavier, 8 for Nano
   GPU_LAYERS_DRAFT=24
   N_BATCH=512
   ```
   
   You can verify GPU is being used when you see lines like:
   ```
   load_tensors: layer X assigned to device CUDA
   ```
   in the startup logs instead of `assigned to device CPU`.
   
   For the best GPU performance:
   - Run `sudo jetson_clocks` to maximize clock speeds
   - Try different values for `N_GPU_LAYERS` (higher values use more GPU memory)
   - Monitor performance with `sudo tegrastats`
   - Use models with Q5_K_M or Q6_K quantization for better GPU compatibility

4. **Monitor Memory Usage**: Use `tegrastats` to monitor memory usage:
   ```bash
   sudo tegrastats
   ```

5. **Reduce Context Window**: If you're still having memory issues, reduce the context window size in your `.env` file:
   ```
   CONTEXT_WINDOW_SIZE=2048
   ```

## Model Selection for Jetson

Different models perform differently on Jetson hardware:

### Recommended for Jetson Orin
- Llama 3 8B (Q5_K_M) - Best balance of quality and performance
- CodeLlama 7B (Q5_K_M) - Excellent for code-related tasks
- Mistral 7B (Q5_K_M) - Good general performance

### Recommended for Jetson Xavier
- Llama 2 7B (Q4_K_M) - Good balance for Xavier
- Phi-2 (Q4_K_M) - Smaller but capable model

### Recommended for Jetson Nano
- TinyLlama (Q4_K_M) - Small enough to run on Nano

You can download these models using our download script:
```bash
poetry run python download_model.py
```

## Troubleshooting

### ChromaDB Migration Issues

If you see an error about deprecated ChromaDB configuration:

```
Error initializing ChromaDB with custom settings: You are using a deprecated configuration of Chroma.
```

This is because ChromaDB has updated its architecture. The simplest solution is to:

1. Delete the existing ChromaDB data:
   ```bash
   rm -rf .codex_data/chroma
   ```

2. Run the Jetson setup script which will configure ChromaDB correctly:
   ```bash
   poetry run python jetson_setup.py
   ```

3. Run the demo with Jetson-optimized settings:
   ```bash
   poetry run python demo.py --jetson
   ```

### Vector Index Out of Bounds Errors

If you're seeing vector index out of bounds errors like:
```
std::vector<_Tp, _Alloc>::reference std::vector<_Tp, _Alloc>::operator[](size_type) [with _Tp = unsigned int...]: Assertion '__n < this->size()' failed.
```

This is related to ChromaDB's HNSW index implementation and embedding generation. To fix this:

1. Delete the existing ChromaDB data:
   ```bash
   rm -rf .codex_data/chroma
   ```

2. Add these settings to your `.env` file:
   ```
   # Embedding model settings
   EMBEDDING_MODEL=paraphrase-MiniLM-L3-v2
   EMBEDDING_BATCH_SIZE=1
   ```

3. Run the demo with very small chunk sizes:
   ```bash
   poetry run python demo.py --code-chunk-size 300 --doc-chunk-size 500
   ```

Our fixes include:
- Using a smaller, more stable embedding model (paraphrase-MiniLM-L3-v2)
- Processing embeddings one at a time to avoid memory issues
- Adding robust error handling for embedding generation
- Increasing the search_ef parameter for better recall

### Out of Memory Errors

If you encounter out-of-memory errors:

1. Reduce chunk sizes further (try 300 for code, 500 for docs)
2. Process smaller codebases or subsets of your codebase
3. Use a smaller model (TinyLlama instead of Llama 2)

### Slow Performance

Jetson devices are less powerful than desktop systems, so expect:

1. Longer ingestion times (be patient during embedding generation)
2. Slower query responses (30 seconds to several minutes)
3. Higher resource usage (CPU, RAM, and GPU if enabled)

### Embedding Model Issues

If you have issues with the embedding model:

1. Try using a different embedding model by setting in your `.env`:
   ```
   EMBEDDING_MODEL=paraphrase-MiniLM-L3-v2
   ```

2. Or disable the embedding model and use simple keyword matching:
   ```
   USE_EMBEDDINGS=false
   ```

### GPU Acceleration Issues

If GPU acceleration isn't working:

1. **Check CUDA Installation**:
   ```bash
   nvcc --version
   ```
   Make sure CUDA is properly installed.

2. **Verify llama-cpp-python CUDA Support**:
   ```bash
   poetry run python -c "import llama_cpp; print('CUDA support:', hasattr(llama_cpp._lib, 'llama_backend_cuda_init'))"
   ```
   Should output `CUDA support: True`

3. **Recompile with Correct CUDA Path**:
   If your CUDA is in a non-standard location, specify the path:
   ```bash
   CUDACXX=/usr/local/cuda-XX.X/bin/nvcc CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall
   ```

4. **Try Different Model Formats**:
   Some quantization formats work better with CUDA:
   - Q5_K_M and Q6_K typically work best with GPU
   - Q4_K_M is a good balance of quality and performance
   - Q2_K and Q3_K may not benefit as much from GPU

5. **Monitor GPU Usage**:
   ```bash
   sudo tegrastats
   ```
   Check if GPU memory is being utilized during inference.

## Supported Jetson Devices

This configuration has been tested on:

- Jetson Orin (AGX, NX)
- Jetson Xavier (AGX, NX)

Older devices like Jetson Nano may struggle with the LLM but can still use the embedding and retrieval functionality.

## Getting Help

If you encounter issues specific to Jetson devices, please file an issue on GitHub with:

1. Your exact Jetson model
2. JetPack/L4T version
3. Complete error message
4. Steps to reproduce 