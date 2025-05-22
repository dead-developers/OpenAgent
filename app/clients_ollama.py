"""
Ollama client integration for OpenAgent using the official Ollama Python library.
This provides a wrapper that makes the Ollama API compatible with OpenAgent's expected format.
"""

import json
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import ollama
from ollama import AsyncClient as OllamaAsyncClient
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage

from app.logger import logger


class OllamaClientWrapper:
    """Wrapper for the official Ollama Python client to make it compatible with OpenAgent."""

    def __init__(self, model: str = "phi4:14b-q4_K_M", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host
        self.chat = self.Chat(self)
    
    class Chat:
        """Chat completions namespace for compatibility with OpenAI client."""

        def __init__(self, client):
            self.client = client
            self.completions = self

        async def create(self, **kwargs) -> Union[ChatCompletion, AsyncIterator[ChatCompletionChunk]]:
            """Create a chat completion. Defers to the parent client's implementation."""
            return await self.client.create_chat_completion(**kwargs)

    async def create_chat_completion(
        self, 
        messages: List[dict], 
        model: Optional[str] = None, 
        max_tokens: int = 4096, 
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ) -> Union[ChatCompletion, AsyncIterator[ChatCompletionChunk]]:
        """
        Create a chat completion using Ollama's API via the official Python client.
        
        Args:
            messages: List of conversation messages
            model: Model name in Ollama (defaults to instance model if not provided)
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream the response
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            ChatCompletion or AsyncIterator[ChatCompletionChunk]
        """
        # Use the model specified in the call, or fall back to the instance model
        model_to_use = model or self.model
        
        # Set up options for the Ollama client
        options = {
            "num_predict": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        try:
            # Use the async client for better performance
            ollama_client = OllamaAsyncClient(host=self.host)
            
            if stream:
                # For streaming responses
                return self._stream_chat_completion(ollama_client, model_to_use, messages, options)
            else:
                # For non-streaming responses
                return await self._request_chat_completion(ollama_client, model_to_use, messages, options)
                
        except Exception as e:
            logger.error(f"Error in Ollama chat completion: {str(e)}")
            raise
    
    async def _request_chat_completion(
        self, 
        client: OllamaAsyncClient, 
        model: str, 
        messages: List[dict],
        options: dict
    ) -> ChatCompletion:
        """Make a non-streaming request to Ollama API."""
        try:
            # Call the Ollama API
            response = await client.chat(
                model=model,
                messages=messages,
                options=options,
                stream=False
            )
            
            # Create a response in OpenAI format
            completion = {
                "id": f"ollama-{model}-{hash(str(messages))}",
                "object": "chat.completion",
                "created": int(response.get("created_at", 0)),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response.get("message", {}).get("content", "")
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": response.get("prompt_eval_count", 0),
                    "completion_tokens": response.get("eval_count", 0),
                    "total_tokens": response.get("prompt_eval_count", 0) + response.get("eval_count", 0)
                }
            }
            
            return ChatCompletion(**completion)
        
        except Exception as e:
            logger.error(f"Error in Ollama chat completion: {str(e)}")
            raise
    
    async def _stream_chat_completion(
        self, 
        client: OllamaAsyncClient, 
        model: str, 
        messages: List[dict],
        options: dict
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Stream a chat completion from Ollama API."""
        try:
            # Call the Ollama API with streaming
            stream = await client.chat(
                model=model,
                messages=messages,
                options=options,
                stream=True
            )
            
            # Process the streaming response
            async for chunk in stream:
                # Create a chunk in OpenAI format
                formatted_chunk = {
                    "id": f"ollama-{model}-{hash(str(messages))}",
                    "object": "chat.completion.chunk",
                    "created": int(chunk.get("created_at", 0)),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": chunk.get("message", {}).get("content", "")
                            },
                            "finish_reason": None if not chunk.get("done", False) else "stop"
                        }
                    ]
                }
                
                yield ChatCompletionChunk(**formatted_chunk)
                
        except Exception as e:
            logger.error(f"Error in Ollama chat completion streaming: {str(e)}")
            raise
