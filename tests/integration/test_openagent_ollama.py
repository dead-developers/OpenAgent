#!/usr/bin/env python3
"""
Test script for OpenAgent with Ollama integration
This script tests the OpenAgent using the Ollama models
"""

import asyncio
import os
import sys
from app.agent.manus import Manus
from app.logger import logger
from app.llm import OllamaLLM
from app.clients import OllamaClient

async def test_openagent_ollama(prompt="What is the capital of France?", model_name="deepseek-r1:14b"):
    """Test OpenAgent with Ollama integration"""
    logger.info(f"Testing OpenAgent with Ollama model: {model_name}")
    
    # Create a custom Ollama LLM instance
    llm = OllamaLLM(model_name)
    
    # Create and initialize Manus agent
    agent = await Manus.create()
    # Replace the default LLM with our Ollama LLM
    agent.llm = llm
    
    try:
        logger.info(f"Processing prompt: '{prompt}'")
        result = await agent.run(prompt)
        logger.info(f"Response: {result}")
        return result
    except Exception as e:
        logger.error(f"Error testing OpenAgent with Ollama: {str(e)}")
        return None
    finally:
        # Ensure agent resources are cleaned up
        await agent.cleanup()

async def main():
    """Main function to run the test"""
    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    else:
        prompt = "What is the capital of France?"
        
    if len(sys.argv) > 2:
        model = sys.argv[2]
    else:
        model = "deepseek-r1:14b"
    
    result = await test_openagent_ollama(prompt, model)
    
    if result:
        logger.info("✅ OpenAgent with Ollama test passed!")
    else:
        logger.error("❌ OpenAgent with Ollama test failed!")

if __name__ == "__main__":
    asyncio.run(main())
