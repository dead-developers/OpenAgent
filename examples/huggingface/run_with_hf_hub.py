import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
dotenv_path = Path(__file__).resolve().parent / '.env'
load_dotenv(dotenv_path if dotenv_path.exists() else None)

from app.agent.manus import Manus
from app.logger import logger
from app.llm import LLM
from app.clients_hf_hub import HuggingFaceHubClient

class HuggingFaceHubLLM(LLM):
    """Custom LLM class for HuggingFace Hub integration using the official huggingface_hub library"""
    
    def __init__(self, config_name="default", llm_config=None):
        # Call the parent class's __init__ method first
        super().__init__(config_name, llm_config)
        
        # Get API key from environment
        api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not api_key:
            raise ValueError("HUGGINGFACE_API_KEY environment variable is not set")
        
        # Create a HuggingFace Hub client
        self.client = HuggingFaceHubClient(
            api_key=api_key,
            model=self.model,
            provider="nebius"
        )
        logger.info(f"Initialized HuggingFace Hub client for model {self.model}")

async def run_with_hf_hub(prompt, model_name=None):
    """Run the Manus agent with HuggingFace Hub LLM"""
    # Create a custom HuggingFace Hub LLM instance
    llm = HuggingFaceHubLLM()
    
    # Override the model if specified
    if model_name:
        llm.model = model_name
        llm.client.model = model_name
    
    # Create and initialize Manus agent
    agent = await Manus.create()
    # Replace the default LLM with our HuggingFace Hub LLM
    agent.llm = llm
    
    try:
        if not prompt.strip():
            logger.warning("Empty prompt provided.")
            return

        logger.warning(f"Processing your request using HuggingFace Hub ({llm.model}): {prompt}")
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
    model = sys.argv[1] if len(sys.argv) > 1 else None
    prompt = sys.argv[2] if len(sys.argv) > 2 else "What is the capital of France?"
    
    # Run the agent with HuggingFace Hub
    result = asyncio.run(run_with_hf_hub(prompt, model))
    print("\nResult:", result)
