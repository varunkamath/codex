# Phi-4 Usage Guide

This guide provides instructions for using the Microsoft Phi-4 model with Codex.

## Overview

Phi-4 is Microsoft's latest powerful small language model, designed to deliver exceptional reasoning capabilities while being efficient enough to run on consumer hardware. It offers impressive performance comparable to much larger models.

Key features:
- Excellent reasoning and instruction-following capabilities
- Multilingual support
- ChatML prompt format
- Optimized for local deployment
- 8-bit quantization for efficient inference
- Apache 2.0 license

## Installation

To download and set up the Phi-4 model:

```bash
# Download the model
python download_model.py

# Select "phi-4" when prompted
```

Note: The model is approximately 4.2 GB in size, so ensure you have sufficient disk space.

## Hardware Requirements

The Phi-4 model requires:

- **RAM**: At least 16GB of system RAM
- **GPU**: For optimal performance, a GPU with at least 8GB VRAM is recommended
- **Storage**: At least 5GB of free disk space

For Jetson devices, this model is suitable for Jetson Orin with good cooling.

## Prompt Format

Phi-4 uses the ChatML prompt format:

```
<|im_start|>system<|im_sep|>
You are a helpful AI assistant.<|im_end|>
<|im_start|>user<|im_sep|>
How can I learn Python?<|im_end|>
<|im_start|>assistant<|im_sep|>
```

Codex automatically handles this format when using the model, so you don't need to format prompts manually.

### Prompt Format Details

The ChatML format has specific requirements for conversation flow:

1. The initial exchange includes:
   - System message wrapped in `<|im_start|>system<|im_sep|>` and `<|im_end|>` tags
   - User message wrapped in `<|im_start|>user<|im_sep|>` and `<|im_end|>` tags
   - Assistant response starts with `<|im_start|>assistant<|im_sep|>`

2. Subsequent exchanges follow this pattern:
   - User message wrapped in `<|im_start|>user<|im_sep|>` and `<|im_end|>` tags
   - Assistant response wrapped in `<|im_start|>assistant<|im_sep|>` and `<|im_end|>` tags

Codex implements this format correctly, including proper handling of conversation history in chat mode.

## Usage

### Basic Query

```bash
python -m codex.main query "How does the authentication system work?" --data-dir .your_codebase_data
```

### Interactive Chat Mode

For the best experience with Phi-4, use the interactive chat mode:

```bash
python -m codex.main chat --data-dir .your_codebase_data
```

This keeps the model loaded between queries and maintains conversation history.

### Environment Variables

You can customize the model's behavior with these environment variables:

- `CONTEXT_WINDOW_SIZE`: Set the context window size (default: 200000)
- `USE_CHAT_HISTORY`: Enable/disable chat history (default: "true")
- `MAX_CHAT_HISTORY`: Maximum number of exchanges to keep (default: 10)

## Performance Tips

1. **GPU Acceleration**: For optimal performance, ensure you have CUDA support:
   ```bash
   CUDACXX=/usr/local/cuda/bin/nvcc CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
   ```

2. **Quantization**: The model uses Q8_0 quantization for a good balance of quality and performance.

3. **Batch Size**: Adjust the batch size based on your hardware:
   ```bash
   python -m codex.main query "Your question" --data-dir .your_codebase_data --batch-size 512
   ```

4. **Memory Management**: If you encounter memory issues:
   - Reduce the context window size
   - Use a smaller batch size
   - Close other memory-intensive applications

## Troubleshooting

### Out of Memory Errors

If you encounter out of memory errors:

1. Reduce the context window size:
   ```bash
   CONTEXT_WINDOW_SIZE=16000 python -m codex.main query "Your question" --data-dir .your_codebase_data
   ```

2. Reduce the number of GPU layers:
   ```bash
   python -m codex.main query "Your question" --data-dir .your_codebase_data --gpu-layers 20
   ```

### Slow Responses

If the model is responding slowly:

1. Increase the batch size if you have sufficient GPU memory
2. Ensure you're using GPU acceleration
3. Consider using a smaller model if your hardware is limited

### Response Format

The Phi-4 model returns responses that may include ChatML formatting tokens. Codex automatically cleans these responses by:

1. Extracting only the content between `<|im_start|>assistant<|im_sep|>` and `<|im_end|>`
2. Removing any remaining ChatML tags
3. Trimming any leading or trailing whitespace

This ensures that you get clean, properly formatted responses without any ChatML formatting artifacts.

If you're using the model outside of Codex, you may need to implement similar post-processing to clean up the responses.

## Additional Resources

- [Phi-4 on Hugging Face](https://huggingface.co/microsoft/Phi-4)
- [Microsoft Phi-4 Blog Post](https://www.microsoft.com/en-us/research/blog/phi-4-advancing-reasoning-with-larger-contexts/)
- [Codex Documentation](./README.md) 