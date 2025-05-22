import asyncio
import os
from pathlib import Path

# Initialize dotenv before other imports to ensure environment variables are available
from dotenv import load_dotenv

# Load environment variables from .env file
dotenv_path = Path(__file__).resolve().parent / '.env'
load_dotenv(dotenv_path if dotenv_path.exists() else None)

from app.agent.manus import Manus
from app.logger import logger


async def test_agent(prompt="What is the capital of France?"):
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
    result = asyncio.run(test_agent())
    print("\nResult:", result)
