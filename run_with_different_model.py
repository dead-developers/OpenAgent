import asyncio
import os
import sys
from app.agent.manus import Manus
from app.logger import logger

async def run_with_different_model(prompt):
    # Override the model to use a different HuggingFace model
    os.environ["OPENAGENT_MODEL"] = "mistralai/Mistral-7B-Instruct-v0.2"
    
    # Create and initialize Manus agent
    agent = await Manus.create()
    try:
        if not prompt.strip():
            logger.warning("Empty prompt provided.")
            return

        logger.warning(f"Processing your request: {prompt}")
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
    result = asyncio.run(run_with_different_model(prompt))
    print("\nResult:", result)
