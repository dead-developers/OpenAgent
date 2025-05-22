# OpenAgent HuggingFace Hub Integration

This directory contains examples for using OpenAgent with HuggingFace Hub API to access various AI models.

## Files

- `run_with_hf_hub.py`: Example script for running OpenAgent with HuggingFace Hub using DeepSeek models

## Configuration

The HuggingFace Hub configuration is defined in `/config/config.toml`. This configuration specifies:

- Model name (e.g., `deepseek-ai/DeepSeek-V3`)
- API endpoint (`https://api-inference.huggingface.co/models`)
- API key (loaded from environment variable `HUGGINGFACE_API_KEY`)
- Model parameters (temperature, max tokens, etc.)

## Running Examples

To run an example with HuggingFace Hub:

```bash
python run_with_hf_hub.py "deepseek-ai/DeepSeek-R1" "Your prompt here"
```

## Requirements

- A HuggingFace API key must be set in the `.env` file
- Internet connection to access the HuggingFace Hub API
- Access permissions for the specified model on HuggingFace
