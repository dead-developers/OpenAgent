#!/usr/bin/env python3
"""
OpenAgent Runner Script

This script provides a simple way to run OpenAgent with different configurations.
It will automatically use the models configured in config/config.toml with fallback
options if the primary model is unavailable.
"""

import asyncio
import os
import sys
from app.agent.manus import Manus
from app.logger import logger

async def run_agent(prompt):
    """Run the OpenAgent with the configured models"""
    # Create and initialize Manus agent
    agent = await Manus.create()
    
    try:
        if not prompt.strip():
            logger.warning("Empty prompt provided.")
            return

        logger.info(f"Processing your request: {prompt}")
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
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is the capital of France?"
    
    # Run the agent
    result = asyncio.run(run_agent(prompt))
    
    # Display the result
    if result:
        print("\nAgent Response:")
        print("-" * 50)
        print(result)
        print("-" * 50)
    else:
        print("\nNo response received from the agent.")
