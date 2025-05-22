# OpenAgent Ollama Integration

This directory contains examples for using OpenAgent with Ollama for local model inference.

## Files

- `run_with_ollama.py`: Example script for running OpenAgent with Ollama using DeepSeek-R1 model
- `run_with_ollama_official.py`: Example script using the official Ollama Python client

## Configuration

The Ollama configuration is defined in `/config/ollama_config.toml`. This configuration specifies:

- Model name (e.g., `deepseek-r1:14b`)
- API endpoint (default: `http://localhost:11434`)
- Model parameters (temperature, max tokens, etc.)

## Running Examples

To run an example with Ollama:

```bash
python run_with_ollama.py "Your prompt here"
```

## Requirements

- Ollama must be installed and running locally
- The specified model must be downloaded via Ollama
  - You can download a model with: `ollama pull deepseek-r1:14b`
