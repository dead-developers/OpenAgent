import asyncio
import os
import sys
from app.agent.manus import Manus
from app.logger import logger
from app.llm import LLM
from app.clients import OllamaClient

class OllamaLLM(LLM):
    """Custom LLM class for Ollama integration"""
    
    def __init__(self, model_name="deepseek-r1:14b"):
        super().__init__("ollama")
        # Override the model name if specified
        self.model = model_name
        # Create an Ollama client
        self.client = OllamaClient(
            api_key="",  # No API key needed for Ollama
            base_url="http://localhost:11434",  # Default Ollama endpoint
            model=model_name  # Set the model in the client
        )
        logger.info(f"Initialized Ollama client for model {self.model}")

async def run_with_ollama(prompt, model_name="deepseek-r1:14b"):
    """Run the Manus agent with Ollama LLM"""
    # Create a custom Ollama LLM instance
    llm = OllamaLLM(model_name)
    
    # Create and initialize Manus agent
    agent = await Manus.create()
    # Replace the default LLM with our Ollama LLM
    agent.llm = llm
    
    try:
        if not prompt.strip():
            logger.warning("Empty prompt provided.")
            return

        logger.warning(f"Processing your request using Ollama ({model_name}): {prompt}")
        result = await agent.run(prompt)
        logger.info("Request processing completed.")
        return result
    except KeyboardInterrupt:
        logger.warning("Operation interrupted.")
    finally:
        # Ensure agent resources are cleaned up before exiting
        await agent.cleanup()

if __name__ == "__main__":
    # Get model name and prompt from command line arguments
    model = sys.argv[1] if len(sys.argv) > 1 else "deepseek-r1:14b"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Tell me a joke about programming"
    
    # Run the agent with Ollama
    result = asyncio.run(run_with_ollama(prompt, model))
    print("\nResult:", result)
