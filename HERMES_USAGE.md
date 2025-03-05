# Using Hermes 3 with Codex

This guide explains how to use the Hermes 3 model with Codex, including its special prompt format and enhanced capabilities.

## What is Hermes 3?

Hermes 3 is a state-of-the-art language model built on top of Meta's Llama 3.1 8B foundation model. It has been fine-tuned by NousResearch to enhance:

- Advanced agentic capabilities
- Better roleplaying and reasoning
- Improved multi-turn conversation
- Long context coherence
- Function calling and structured outputs
- Improved code generation

## Getting Started

### 1. Download the Hermes 3 Model

```bash
poetry run python download_model.py
```

Select `hermes-3-llama-3.1-8b` when prompted.

### 2. Switch to the Hermes Model

If you have multiple models downloaded, you can switch between them using:

```bash
poetry run python switch_model.py
```

Select the Hermes 3 model from the list.

### 3. Optimize for GPU Acceleration (Jetson Only)

If you're using a Jetson device, optimize for GPU acceleration:

```bash
poetry run python jetson_gpu_optimize.py
```

### 4. Run the Demo

```bash
poetry run python demo.py
```

For Jetson devices:

```bash
poetry run python demo.py --jetson
```

## Interactive Chat Mode

Codex now includes an interactive chat mode that keeps the model loaded between queries for faster response times. This is especially useful when you want to have a conversation about your codebase without reinitializing the model for each query.

### Starting Interactive Chat

```bash
poetry run python -m codex.main chat -d .your_codebase_data
```

This will:
1. Load your codebase context from the specified data directory
2. Initialize the model once (with a 200,000 token context window by default)
3. Start an interactive chat session where you can ask multiple questions

### Options

- `-d, --data-dir`: Directory with processed codebase data (default: `.codex_data`)
- `-m, --model`: Model to use (default: `gpt-3.5-turbo`, but will use your local model if `LOCAL_MODEL_PATH` is set)
- `-c, --context-window`: Size of the context window (default: `200000`)

### Example

```bash
poetry run python -m codex.main chat -d .my_project_data -c 100000
```

### Chat Commands

While in chat mode, you can use these commands:
- `exit`, `quit`, `q`, or `bye`: Exit the chat session
- Empty line: Will be ignored

### Environment Variables

You can customize the chat behavior with these environment variables:
- `USE_CHAT_HISTORY`: Set to "false" to disable chat history (default: "true")
- `MAX_CHAT_HISTORY`: Maximum number of exchanges to keep in history (default: 10)
- `CONTEXT_WINDOW_SIZE`: Size of the context window (default: 200000)

## ChatML Prompt Format

Hermes 3 uses the ChatML prompt format, which is different from the standard format used by other models. The format looks like this:

```
<|im_start|>system
You are Hermes 3, a helpful AI assistant.
<|im_end|>
<|im_start|>user
Hello, who are you?
<|im_end|>
<|im_start|>assistant
Hi there! I'm Hermes 3, an AI assistant built on the Llama 3.1 architecture...
<|im_end|>
```

**Don't worry!** Codex has been updated to automatically detect Hermes models and use the correct prompt format. You don't need to make any changes to your workflow.

## Advanced Capabilities

### Function Calling

Hermes 3 has enhanced capabilities for function calling, which allows it to interact with external tools and APIs. This is particularly useful for tasks like:

- Retrieving information from databases
- Calling external APIs
- Performing calculations
- Executing code

### Structured Outputs

Hermes 3 can generate structured outputs in JSON format, making it easier to parse and use the results in your applications.

## Troubleshooting

### Model Not Responding Correctly

If the model isn't responding correctly, it might be due to the prompt format. Check that:

1. You're using the latest version of Codex that supports ChatML format
2. The model is correctly detected as a Hermes model

You can verify this by checking the logs for:
```
Detected Hermes model, using ChatML format
```

### Performance Issues

If you're experiencing performance issues:

1. For Jetson devices, make sure GPU acceleration is enabled
2. Try reducing the context window size if you're running out of memory
3. Use the Q5_K_M quantization for the best balance of quality and performance

## Further Resources

- [Hermes 3 Technical Report](https://arxiv.org/abs/2408.11857)
- [Hermes Function Calling GitHub](https://github.com/NousResearch/Hermes-Function-Calling)
- [Hugging Face Model Page](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B) 