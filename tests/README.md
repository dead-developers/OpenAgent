# OpenAgent Tests

This directory contains test scripts for validating OpenAgent functionality.

## Directory Structure

- `integration/`: Integration tests for different model providers
- `unit/`: Unit tests for individual components

## Running Tests

To run integration tests, navigate to the integration directory and execute the desired test script:

```bash
cd integration
python test_openrouter.py  # Test OpenRouter integration
python test_ollama_integration.py  # Test Ollama integration
```

These tests validate that OpenAgent can properly communicate with different model providers and handle responses correctly.
