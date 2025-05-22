#!/usr/bin/env python3
"""
Test script for Ollama integration with OpenAgent
This script tests the basic functionality of the Ollama client with OpenAgent
"""

import asyncio
import os
from app.clients import OllamaClient
from app.logger import logger

async def test_ollama_connection(model_name="deepseek-r1:14b"):
    """Test basic connection to Ollama and model availability"""
    logger.info(f"Testing Ollama connection with model: {model_name}")
    
    # Create Ollama client
    client = OllamaClient(
        api_key="",  # No API key needed for Ollama
        base_url="http://localhost:11434",  # Default Ollama endpoint
        model=model_name
    )
    
    # Test simple completion
    test_prompt = "What is the capital of France?"
    logger.info(f"Sending test prompt: '{test_prompt}'")
    
    try:
        # Create a simple chat completion
        messages = [{"role": "user", "content": test_prompt}]
        response = await client.create_chat_completion(
            messages=messages,
            model=model_name,
            max_tokens=100,
            temperature=0.7,
            stream=False
        )
        
        # Display response
        if response and "choices" in response and len(response["choices"]) > 0:
            content = response["choices"][0]["message"]["content"]
            logger.info(f"Received response: {content}")
            return True
        else:
            logger.error(f"Unexpected response format: {response}")
            return False
            
    except Exception as e:
        logger.error(f"Error testing Ollama connection: {str(e)}")
        return False

async def main():
    """Main function to run tests"""
    # Test default model
    success = await test_ollama_connection()
    
    if success:
        logger.info("✅ Ollama integration test passed!")
    else:
        logger.error("❌ Ollama integration test failed!")

if __name__ == "__main__":
    asyncio.run(main())
