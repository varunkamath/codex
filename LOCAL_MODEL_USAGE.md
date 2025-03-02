# Using Local Models with Codex

This guide explains how to set up and use Codex with local language models instead of OpenAI, which is especially useful for:
- Working offline
- Handling private codebases without uploading content to external APIs
- Avoiding API costs

## Quick Start

The easiest way to get started with a local model is to use our automatic downloader:

```bash
# Install required packages
poetry add requests tqdm

# Run the downloader
poetry run python download_model.py
```

This script will:
1. Show you available models to choose from
2. Download your selected model
3. Update your `.env` file automatically
4. Provide next steps to run the demo

## Manual Setup Process

If you prefer to set up manually, follow these steps:

### Step 1: Install Dependencies

```bash
# Using Poetry (recommended)
poetry install
poetry add langchain-community llama-cpp-python

# Or with pip
pip install -r requirements.txt
pip install langchain-community llama-cpp-python
```

### Step 2: Download a Model

Codex uses models in GGUF format (compatible with llama.cpp). Here are some recommended options:

| Model | Size | Quality | Speed | Link |
|-------|------|---------|-------|------|
| Llama 2 7B Chat (Q4_K_M) | 4.1 GB | Good | Medium | [Download](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf) |
| Llama 2 7B Chat (Q2_K) | 2.9 GB | Lower | Faster | [Download](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q2_K.gguf) |
| TinyLlama 1.1B | 0.7 GB | Basic | Very Fast | [Download](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf) |

Download your chosen model and place it in a `models` directory:

```bash
# Create models directory
mkdir -p models

# Download a model (example with Llama 2)
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf -O models/llama-2-7b-chat.Q4_K_M.gguf
```

### Step 3: Configure Codex

Update your `.env` file to use the local model:

```bash
# OpenAI API Configuration
# Comment out the API key to force using the local model
# OPENAI_API_KEY=your_api_key_here

# Local Model Configuration
LOCAL_MODEL_PATH=models/llama-2-7b-chat.Q4_K_M.gguf

# Avoid tokenizers parallelism warning
TOKENIZERS_PARALLELISM=false
```

### Step 4: Run the Demo

Now run the demo as usual:

```bash
# Using Poetry
poetry run python demo.py

# Or directly
python demo.py
```

## Performance Considerations

### Hardware Requirements

Local models have different hardware requirements:

- **Llama 2 7B (Q4_K_M)**: 8GB+ RAM recommended
- **Llama 2 7B (Q2_K)**: 6GB+ RAM recommended
- **TinyLlama 1.1B**: 2GB+ RAM recommended

### Speed

Local models on CPU are significantly slower than OpenAI's API. Expect:

- First-time loading: 30-60 seconds
- Response generation: 30 seconds to 2 minutes per query (depending on model size and hardware)

### GPU Acceleration

If you have a compatible GPU, you can dramatically speed up inference:

```bash
# For NVIDIA GPUs (CUDA)
pip uninstall llama-cpp-python
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python

# For Apple Silicon (Metal)
pip uninstall llama-cpp-python
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python-metal
```

## Troubleshooting

### Common Issues

1. **Memory Errors**: If you get out-of-memory errors:
   - Try a smaller model (Q2_K instead of Q4_K_M, or TinyLlama)
   - Reduce the context window in `query_engine.py` (change `n_ctx=4096` to `n_ctx=2048`)

2. **ImportError: cannot import name 'LlamaCpp'**:
   - Ensure you've installed langchain-community: `poetry add langchain-community`

3. **Model Not Found**:
   - Check that the path in your .env file matches where you saved the model
   - Ensure the model file exists and you have read permissions

4. **Slow Responses**:
   - This is normal for local models on CPU
   - Try GPU acceleration if available (see above)
   - Consider a smaller model for faster responses

### Logs and Debugging

To see more detailed logs when running Codex:

```bash
# Set environment variable for verbose logging
export LOGLEVEL=DEBUG

# Run with logging
poetry run python demo.py
```

## Advanced Configuration

You can customize the local model behavior by editing the LlamaCpp parameters in `codex/llm/query_engine.py`:

```python
self.llm = LlamaCpp(
    model_path=model_path,
    temperature=0.1,  # Controls randomness (0.0-1.0)
    n_ctx=4096,       # Context window size
    # Additional options:
    # n_gpu_layers=-1,  # Number of layers to offload to GPU (-1 for all)
    # n_batch=512,      # Batch size for prompt processing
    # verbose=True,     # Enable verbose mode for debugging
)
```

## Contributing

If you improve local model support, please submit a pull request!

Potential improvements:
- Better model download tools
- GPU acceleration setup scripts
- Support for more model architectures
- Performance optimizations 