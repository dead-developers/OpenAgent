import json
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import httpx
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage

from app.clients import BaseClient
from app.logger import logger


class OllamaClient(BaseClient):
    """Client for Ollama API."""

    def __init__(self, api_key: str = "", base_url: str = "http://localhost:11434", **kwargs):
        headers = {
            "Content-Type": "application/json"
        }
        super().__init__(api_key, base_url, headers=headers, **kwargs)
        self.chat = self.Chat(self)
    
    def _format_messages_for_ollama(self, messages: List[dict]) -> List[dict]:
        """Format OpenAI-style messages for Ollama API."""
        # Ollama expects messages in the same format as OpenAI
        return messages

    async def create_chat_completion(
        self, 
        messages: List[dict], 
        model: str, 
        max_tokens: int = 4096, 
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ) -> Union[ChatCompletion, AsyncIterator[ChatCompletionChunk]]:
        """
        Create a chat completion using Ollama's API.
        
        Args:
            messages: List of conversation messages
            model: Model name in Ollama
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream the response
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            ChatCompletion or AsyncIterator[ChatCompletionChunk]
        """
        endpoint = f"{self.base_url}/api/chat"
        
        # Format messages for Ollama
        formatted_messages = self._format_messages_for_ollama(messages)
        
        payload = {
            "model": model,
            "messages": formatted_messages,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                **kwargs
            },
            "stream": stream
        }
        
        if stream:
            return self._stream_chat_completion(endpoint, payload)
        else:
            return await self._request_chat_completion(endpoint, payload)
    
    async def _request_chat_completion(self, endpoint: str, payload: dict) -> ChatCompletion:
        """Make a non-streaming request to Ollama API."""
        try:
            response = await self.async_client.post(endpoint, json=payload)
            response.raise_for_status()
            
            data = response.json()
            
            # Create a response in OpenAI format
            completion = {
                "id": f"ollama-{payload.get('model', 'unknown')}-{hash(str(payload.get('messages', [])))}",
                "object": "chat.completion",
                "created": int(response.elapsed.total_seconds()),
                "model": payload.get('model', 'unknown'),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": data.get("message", {}).get("content", "")
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                }
            }
            
            return ChatCompletion(**completion)
        
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error in Ollama chat completion: {str(e)}")
            raise
    
    async def _stream_chat_completion(self, endpoint: str, payload: dict) -> AsyncIterator[ChatCompletionChunk]:
        """Stream a chat completion from Ollama API."""
        payload["stream"] = True
        
        try:
            async with self.async_client.stream("POST", endpoint, json=payload) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    
                    try:
                        data = json.loads(line)
                        
                        # Create a chunk in OpenAI format
                        chunk = {
                            "id": f"ollama-{payload.get('model', 'unknown')}-{hash(str(payload.get('messages', [])))}",
                            "object": "chat.completion.chunk",
                            "created": int(response.elapsed.total_seconds()),
                            "model": payload.get('model', 'unknown'),
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "content": data.get("message", {}).get("content", "")
                                    },
                                    "finish_reason": None if not data.get("done", False) else "stop"
                                }
                            ]
                        }
                        
                        yield ChatCompletionChunk(**chunk)
                        
                        if data.get("done", False):
                            break
                            
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse streaming response line: {line}")
        
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama API streaming error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error in Ollama chat completion streaming: {str(e)}")
            raise
