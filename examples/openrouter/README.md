# OpenAgent OpenRouter Integration

This directory contains examples for using OpenAgent with OpenRouter API to access various AI models.

## Files

- `run_with_openrouter.py`: Example script for running OpenAgent with OpenRouter using DeepSeek models

## Configuration

The OpenRouter configuration is defined in `/config/config.toml`. This configuration specifies:

- Model name (e.g., `deepseek/deepseek-chat-v3-0324:free`)
- API endpoint (`https://api.openrouter.ai/api/v1`)
- API key (loaded from environment variable `OPENROUTER_API_KEY`)
- Model parameters (temperature, max tokens, etc.)

## Running Examples

To run an example with OpenRouter:

```bash
python run_with_openrouter.py "Your prompt here"
```

## Requirements

- An OpenRouter API key must be set in the `.env` file
- Internet connection to access the OpenRouter API
