import asyncio
import os
import httpx
import json
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
dotenv_path = Path(__file__).resolve().parent / '.env'
load_dotenv(dotenv_path if dotenv_path.exists() else None)

# Get API key from environment
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

async def test_openrouter_connection():
    """Test direct connection to OpenRouter API"""
    print(f"Testing OpenRouter API connection...")
    
    # Set up headers
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/dead-developers/OpenAgent",
        "X-Title": "OpenAgent API Test"
    }
    
    # Set up the request payload
    payload = {
        "model": "deepseek/deepseek-chat:free",  # Free model on OpenRouter
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "max_tokens": 1024,
        "temperature": 0.7
    }
    
    # Make the request
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            print(f"Sending request to OpenRouter API...")
            print(f"API Key: {OPENROUTER_API_KEY[:5]}...{OPENROUTER_API_KEY[-5:]}")
            response = await client.post(
                "https://api.openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            # Check response
            if response.status_code == 200:
                data = response.json()
                print(f"Success! Response received:")
                print(f"Model: {data.get('model', 'unknown')}")
                if data.get('choices') and len(data['choices']) > 0:
                    content = data['choices'][0].get('message', {}).get('content', '')
                    print(f"Response: {content}")
                return True
            else:
                print(f"Error: Status code {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"Exception during API call: {str(e)}")
            return False

async def main():
    success = await test_openrouter_connection()
    if success:
        print("OpenRouter API is working correctly!")
    else:
        print("Failed to connect to OpenRouter API. Please check your API key and network connection.")

if __name__ == "__main__":
    asyncio.run(main())
