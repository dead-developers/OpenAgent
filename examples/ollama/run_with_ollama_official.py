import asyncio
import os
import sys
from app.agent.manus import Manus
from app.logger import logger
from app.llm import LLM
from app.clients_ollama import OllamaClientWrapper

class OllamaLLM(LLM):
    """Custom LLM class for Ollama integration using the official Ollama Python library"""
    
    def __init__(self, config_name="default", llm_config=None, model_name="phi4:14b-q4_K_M"):
        # Call the parent class's __init__ method first
        super().__init__(config_name, llm_config)
        # Override the model name if specified
        self.model = model_name
        # Create an Ollama client wrapper
        self.client = OllamaClientWrapper(model=model_name)
        logger.info(f"Initialized Ollama client for model {self.model}")

async def run_with_ollama(prompt, model_name="phi4:14b-q4_K_M"):
    """Run the Manus agent with Ollama LLM"""
    # Create a custom Ollama LLM instance
    llm = OllamaLLM(model_name=model_name)
    
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
    model = sys.argv[1] if len(sys.argv) > 1 else "phi4:14b-q4_K_M"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "What is the capital of France?"
    
    # Run the agent with Ollama
    result = asyncio.run(run_with_ollama(prompt, model))
    print("\nResult:", result)
