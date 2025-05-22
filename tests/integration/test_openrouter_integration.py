#!/usr/bin/env python3
"""
Test script for OpenRouter integration with OpenAgent
This script tests the OpenAgent using the DeepSeek models from OpenRouter
"""

import asyncio
import os
import sys
from app.agent.manus import Manus
from app.logger import logger
from app.llm import LLM
from app.clients import OpenRouterClient

class OpenRouterLLM(LLM):
    """Custom LLM class for OpenRouter integration"""
    
    def __init__(self, config_name="default", llm_config=None, model_name="deepseek/deepseek-chat-v3-0324:free"):
        super().__init__("openrouter", config_name, llm_config)
        # Override the model name if specified
        self.model = model_name
        # Get API key from environment
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")
        
        # Create an OpenRouter client
        self.client = OpenRouterClient(
            api_key=api_key,
            base_url="https://api.openrouter.ai/api/v1",
            model=model_name
        )
        logger.info(f"Initialized OpenRouter client for model {self.model}")

async def test_openrouter(prompt="What is the capital of France?", model_name="deepseek/deepseek-chat-v3-0324:free"):
    """Test OpenAgent with OpenRouter integration"""
    logger.info(f"Testing OpenAgent with OpenRouter model: {model_name}")
    
    # Create a custom OpenRouter LLM instance
    llm = OpenRouterLLM(model_name=model_name)
    
    # Create and initialize Manus agent
    agent = await Manus.create()
    # Replace the default LLM with our OpenRouter LLM
    agent.llm = llm
    
    try:
        logger.info(f"Processing prompt: '{prompt}'")
        result = await agent.run(prompt)
        logger.info(f"Response: {result}")
        return result
    except Exception as e:
        logger.error(f"Error testing OpenAgent with OpenRouter: {str(e)}")
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
        model = "deepseek/deepseek-chat-v3-0324:free"
    
    result = await test_openrouter(prompt, model)
    
    if result:
        logger.info("✅ OpenAgent with OpenRouter test passed!")
    else:
        logger.error("❌ OpenAgent with OpenRouter test failed!")

if __name__ == "__main__":
    asyncio.run(main())
