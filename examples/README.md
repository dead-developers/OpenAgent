# OpenAgent Examples

This directory contains example scripts demonstrating how to use OpenAgent with different model providers.

## Directory Structure

- `ollama/`: Examples for using OpenAgent with Ollama (local inference)
- `openrouter/`: Examples for using OpenAgent with OpenRouter API
- `huggingface/`: Examples for using OpenAgent with HuggingFace Hub API

## Running Examples

To run an example, navigate to the specific provider directory and execute the desired script:

```bash
# For Ollama examples
cd ollama
python run_with_ollama.py "Your prompt here"

# For OpenRouter examples
cd openrouter
python run_with_openrouter.py "Your prompt here"

# For HuggingFace Hub examples
cd huggingface
python run_with_hf_hub.py "Your prompt here"
```

Each example demonstrates how to configure and use OpenAgent with a specific model provider.
