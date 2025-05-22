import asyncio
import os
import sys
from app.agent.manus import Manus
from app.logger import logger
from app.llm import LLM

async def run_with_openrouter(prompt):
    # Force the use of the OpenRouter configuration
    llm = LLM("fallback")  # Use the fallback configuration explicitly
    
    # Create and initialize Manus agent with the OpenRouter LLM
    agent = await Manus.create()
    agent.llm = llm  # Replace the default LLM with our OpenRouter LLM
    
    try:
        if not prompt.strip():
            logger.warning("Empty prompt provided.")
            return

        logger.warning(f"Processing your request using OpenRouter: {prompt}")
        result = await agent.run(prompt)
        logger.info("Request processing completed.")
        return result
    except KeyboardInterrupt:
        logger.warning("Operation interrupted.")
    finally:
        # Ensure agent resources are cleaned up before exiting
        await agent.cleanup()

if __name__ == "__main__":
    # Get prompt from command line arguments or use a default
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Tell me a joke about programming"
    result = asyncio.run(run_with_openrouter(prompt))
    print("\nResult:", result)
