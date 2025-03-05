# Mistral Small 24B Usage Guide

This guide provides instructions for using the Mistral Small 24B Instruct model with Codex.

## Overview

Mistral Small 24B Instruct (2501) is a powerful 24B parameter model that sets a new benchmark in the "small" Large Language Models category below 70B. It offers state-of-the-art capabilities comparable to larger models while being more efficient to run.

Key features:
- 24B parameters with exceptional "knowledge-density"
- Multilingual support for dozens of languages
- Advanced reasoning and conversational capabilities
- 32k context window
- V7-Tekken prompt format
- Apache 2.0 license

## Installation

To download and set up the Mistral Small 24B model:

```bash
# Download the model
python download_model.py

# Select "mistral-small-24b" when prompted
```

Note: The model is approximately 14.2 GB in size, so ensure you have sufficient disk space.

## Hardware Requirements

The Mistral Small 24B model requires more resources than smaller models:

- **RAM**: At least 24GB of system RAM
- **GPU**: For optimal performance, a GPU with at least 16GB VRAM is recommended
- **Storage**: At least 15GB of free disk space

For Jetson devices, this model is best suited for Jetson Orin with good cooling.

## Prompt Format

Mistral Small 24B uses the V7-Tekken prompt format:

```
<s>[SYSTEM_PROMPT]<system prompt>[/SYSTEM_PROMPT][INST]<user message>[/INST]<assistant response></s>[INST]<user message>[/INST]
```

Codex automatically handles this format when using the model, so you don't need to format prompts manually.

### Prompt Format Details

The V7-Tekken format has specific requirements for conversation flow:

1. The initial exchange includes:
   - System prompt wrapped in `[SYSTEM_PROMPT]` tags
   - User message wrapped in `[INST]` tags
   - Assistant response (no special tags)

2. Subsequent exchanges follow this pattern:
   - End previous exchange with `</s>`
   - User message wrapped in `[INST]` tags
   - Assistant response (no special tags)

Codex implements this format correctly, including proper handling of conversation history in chat mode.

## Usage

### Basic Query

```bash
python -m codex.main query "How does the authentication system work?" --data-dir .your_codebase_data
```

### Interactive Chat Mode

For the best experience with Mistral Small 24B, use the interactive chat mode:

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

2. **Quantization**: The model uses Q6_K quantization for a good balance of quality and performance.

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

### Prompt Format Issues

If you encounter issues with the model's responses, such as:
- Empty responses
- Strange formatting or symbols in the output
- Warning messages about duplicate tokens

Try these solutions:

1. Update to the latest version of Codex, which includes fixes for Mistral Small 24B prompt formatting
2. If using the model programmatically, ensure you're following the V7-Tekken format exactly
3. For chat applications, make sure conversation history is formatted correctly with proper `</s>` tokens between exchanges
4. If you see warnings about duplicate `<s>` tokens, try adding a space before `<s>` in your prompt

## Response Format

The Mistral Small 24B model returns responses that may include formatting tokens from the V7-Tekken format. Codex automatically cleans these responses by:

1. Extracting only the content after the last `[/INST]` tag
2. Removing any trailing `</s>` tokens
3. Removing any trailing `[INST]` tokens (which might be part of the next query format)
4. Trimming any leading or trailing whitespace

This ensures that you get clean, properly formatted responses without any V7-Tekken formatting artifacts.

If you're using the model outside of Codex, you may need to implement similar post-processing to clean up the responses.

## Additional Resources

- [Mistral Small 24B on Hugging Face](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501)
- [Mistral AI Blog Post](https://mistral.ai/news/mistral-small/)
- [Codex Documentation](./README.md) 