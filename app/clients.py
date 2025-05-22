import json
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import httpx
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage

from app.logger import logger


class BaseClient:
    """Base class for API clients."""

    def __init__(
        self, api_key: str, base_url: str, timeout: int = 60, headers: Dict[str, str] = None
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.headers = headers or {}
        self.async_client = httpx.AsyncClient(timeout=timeout, headers=self.headers)

    async def close(self):
        """Close the async client."""
        await self.async_client.aclose()

    class Chat:
        """Chat completions namespace for compatibility with OpenAI client."""

        def __init__(self, client):
            self.client = client
            self.completions = self

        async def create(self, **kwargs) -> Union[ChatCompletion, AsyncIterator[ChatCompletionChunk]]:
            """Create a chat completion. Defers to the parent client's implementation."""
            return await self.client.create_chat_completion(**kwargs)


class HuggingFaceClient(BaseClient):
    """Client for HuggingFace Inference API."""

    def __init__(self, api_key: str, base_url: str = "https://api.huggingface.co/models", **kwargs):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        super().__init__(api_key, base_url, headers=headers, **kwargs)
        self.chat = self.Chat(self)
    
    def _format_messages_for_hf(self, messages: List[dict]) -> str:
        """Format OpenAI-style messages for HuggingFace API."""
        formatted_prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            
            if role == "system":
                formatted_prompt += f"<|system|>\n{content}\n"
            elif role == "user":
                formatted_prompt += f"<|user|>\n{content}\n"
            elif role == "assistant":
                formatted_prompt += f"<|assistant|>\n{content}\n"
            else:
                # For other roles, include the role name
                formatted_prompt += f"<|{role}|>\n{content}\n"
        
        # Add assistant prefix for the response
        formatted_prompt += "<|assistant|>\n"
        return formatted_prompt

    async def create_chat_completion(
        self, 
        messages: List[dict], 
        model: str, 
        max_tokens: int = 1024, 
        temperature: float = 1.0,
        stream: bool = False,
        **kwargs
    ) -> Union[ChatCompletion, AsyncIterator[ChatCompletionChunk]]:
        """
        Create a chat completion using HuggingFace's API.
        
        Args:
            messages: List of conversation messages
            model: Model ID on HuggingFace
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream the response
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            ChatCompletion or AsyncIterator[ChatCompletionChunk]
        """
        endpoint = f"{self.base_url}/{model}"
        
        # Format messages for HuggingFace
        formatted_prompt = self._format_messages_for_hf(messages)
        
        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "return_full_text": False,
                **kwargs
            }
        }
        
        if stream:
            return self._stream_chat_completion(endpoint, payload, messages)
        else:
            return await self._request_chat_completion(endpoint, payload, messages)
    
    async def _request_chat_completion(self, endpoint: str, payload: dict, messages: List[dict]) -> ChatCompletion:
        """Make a non-streaming request to HuggingFace API."""
        try:
            response = await self.async_client.post(endpoint, json=payload)
            response.raise_for_status()
            
            data = response.json()
            
            # HuggingFace API returns differently structured responses, normalize to OpenAI format
            if isinstance(data, list) and len(data) > 0:
                generated_text = data[0].get("generated_text", "")
            else:
                generated_text = data.get("generated_text", "")
            
            # Create a response in OpenAI format
            completion = {
                "id": f"hf-{model}-{hash(str(messages))}",
                "object": "chat.completion",
                "created": int(response.elapsed.total_seconds()),
                "model": payload.get("model", "unknown"),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": generated_text
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": len(str(messages)),
                    "completion_tokens": len(generated_text),
                    "total_tokens": len(str(messages)) + len(generated_text)
                }
            }
            
            return ChatCompletion(**completion)
        
        except httpx.HTTPStatusError as e:
            logger.error(f"HuggingFace API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error in HuggingFace chat completion: {str(e)}")
            raise
    
    async def _stream_chat_completion(self, endpoint: str, payload: dict, messages: List[dict]) -> AsyncIterator[ChatCompletionChunk]:
        """Stream a chat completion from HuggingFace API."""
        payload["stream"] = True
        
        try:
            async with self.async_client.stream("POST", endpoint, json=payload) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    
                    try:
                        data = json.loads(line)
                        if "token" in data:
                            token = data["token"]["text"]
                            
                            # Create a chunk in OpenAI format
                            chunk = {
                                "id": f"hf-{payload.get('model', 'unknown')}-{hash(str(messages))}",
                                "object": "chat.completion.chunk",
                                "created": int(response.elapsed.total_seconds()),
                                "model": payload.get("model", "unknown"),
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "content": token
                                        },
                                        "finish_reason": None
                                    }
                                ]
                            }
                            
                            yield ChatCompletionChunk(**chunk)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse streaming response line: {line}")
        
        except httpx.HTTPStatusError as e:
            logger.error(f"HuggingFace API streaming error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error in HuggingFace chat completion streaming: {str(e)}")
            raise


class OpenRouterClient(BaseClient):
    """Client for OpenRouter API."""

    def __init__(self, api_key: str, base_url: str = "https://api.openrouter.ai/api/v1", **kwargs):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": kwargs.pop("http_referer", "https://github.com/dead-developers/OpenAgent"),
            "X-Title": kwargs.pop("x_title", "OpenAgent")
        }
        super().__init__(api_key, base_url, headers=headers, **kwargs)
        self.chat = self.Chat(self)
    
    async def create_chat_completion(
        self, 
        messages: List[dict], 
        model: str, 
        max_tokens: int = 1024, 
        temperature: float = 1.0,
        stream: bool = False,
        **kwargs
    ) -> Union[ChatCompletion, AsyncIterator[ChatCompletionChunk]]:
        """
        Create a chat completion using OpenRouter's API.
        
        Args:
            messages: List of conversation messages (OpenAI format, compatible with OpenRouter)
            model: Model ID on OpenRouter
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream the response
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            ChatCompletion or AsyncIterator[ChatCompletionChunk]
        """
        endpoint = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
            **kwargs
        }
        
        if stream:
            return self._stream_chat_completion(endpoint, payload)
        else:
            return await self._request_chat_completion(endpoint, payload)
    
    async def _request_chat_completion(self, endpoint: str, payload: dict) -> ChatCompletion:
        """Make a non-streaming request to OpenRouter API."""
        try:
            response = await self.async_client.post(endpoint, json=payload)
            response.raise_for_status()
            
            data = response.json()
            
            # OpenRouter API returns OpenAI-compatible format, so we can directly convert
            return ChatCompletion(**data)
        
        except httpx.HTTPStatusError as e:
            logger.error(f"OpenRouter API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error in OpenRouter chat completion: {str(e)}")
            raise
    
    async def _stream_chat_completion(self, endpoint: str, payload: dict) -> AsyncIterator[ChatCompletionChunk]:
        """Stream a chat completion from OpenRouter API."""
        try:
            async with self.async_client.stream("POST", endpoint, json=payload) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if not line or line.strip() == "":
                        continue
                    
                    if line.startswith("data: "):
                        line = line[6:]  # Remove "data: " prefix
                    
                    if line == "[DONE]":
                        break
                    
                    try:
                        data = json.loads(line)
                        yield ChatCompletionChunk(**data)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse streaming response line: {line}")
        
        except httpx.HTTPStatusError as e:
            logger.error(f"OpenRouter API streaming error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error in OpenRouter chat completion streaming: {str(e)}")
            raise


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


class GoogleAIClient(BaseClient):
    """Client for Google AI API (Gemini models)."""

    def __init__(self, api_key: str, base_url: str = "https://generativelanguage.googleapis.com", **kwargs):
        headers = {
            "Content-Type": "application/json"
        }
        super().__init__(api_key, base_url, headers=headers, **kwargs)
        self.chat = self.Chat(self)
    
    def _format_messages_for_gemini(self, messages: List[dict]) -> List[dict]:
        """Convert OpenAI-style messages to Gemini API format."""
        gemini_messages = []
        
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            
            if role == "system":
                # Gemini doesn't have system messages, prepend to first user message
                gemini_messages.append({
                    "role": "user",
                    "parts": [{"text": f"System: {content}"}]
                })
            elif role == "user":
                gemini_messages.append({
                    "role": "user",
                    "parts": [{"text": content if isinstance(content, str) else json.dumps(content)}]
                })
            elif role == "assistant":
                gemini_messages.append({
                    "role": "model",
                    "parts": [{"text": content if isinstance(content, str) else json.dumps(content)}]
                })
        
        return gemini_messages
    
    async def create_chat_completion(
        self, 
        messages: List[dict], 
        model: str, 
        max_tokens: int = 1024, 
        temperature: float = 1.0,
        stream: bool = False,
        **kwargs
    ) -> Union[ChatCompletion, AsyncIterator[ChatCompletionChunk]]:
        """
        Create a chat completion using Google AI API.
        
        Args:
            messages: List of conversation messages
            model: Google AI model name
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream the response
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            ChatCompletion or AsyncIterator[ChatCompletionChunk]
        """
        # Gemini API endpoint
        endpoint = f"{self.base_url}/v1beta/models/{model}:generateContent?key={self.api_key}"
        
        # Format messages for Gemini
        gemini_messages = self._format_messages_for_gemini(messages)
        
        payload = {
            "contents": gemini_messages,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            }
        }
        
        if stream:
            payload["stream"] = True
            return self._stream_chat_completion(endpoint, payload, messages, model)
        else:
            return await self._request_chat_completion(endpoint, payload, messages, model)
    
    async def _request_chat_completion(self, endpoint: str, payload: dict, original_messages: List[dict], model: str) -> ChatCompletion:
        """Make a non-streaming request to Google AI API."""
        try:
            response = await self.async_client.post(endpoint, json=payload)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract content from Gemini response format
            if "candidates" in data and len(data["candidates"]) > 0:
                candidate = data["candidates"][0]
                content = ""
                
                # Extract text from parts
                if "content" in candidate and "parts" in candidate["content"]:
                    for part in candidate["content"]["parts"]:
                        if "text" in part:
                            content += part["text"]
            else:
                content = "No response generated."
            
            # Create a response in OpenAI format
            completion = {
                "id": f"gemini-{model}-{hash(str(original_messages))}",
                "object": "chat.completion",
                "created": int(response.elapsed.total_seconds()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": len(str(original_messages)),
                    "completion_tokens": len(content),
                    "total_tokens": len(str(original_messages)) + len(content)
                }
            }
            
            return ChatCompletion(**completion)
        
        except httpx.HTTPStatusError as e:
            logger.error(f"Google AI API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error in Google AI chat completion: {str(e)}")
            raise
    
    async def _stream_chat_completion(self, endpoint: str, payload: dict, original_messages: List[dict], model: str) -> AsyncIterator[ChatCompletionChunk]:
        """Stream a chat completion from Google AI API."""
        try:
            async with self.async_client.stream("POST", endpoint, json=payload) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if not line or line.strip() == "":
                        continue
                    
                    try:
                        data = json.loads(line)
                        
                        # Extract the text from Gemini's streaming response
                        if "candidates" in data and len(data["candidates"]) > 0:
                            candidate = data["candidates"][0]
                            if "content" in candidate and "parts" in candidate["content"]:
                                for part in candidate["content"]["parts"]:
                                    if "text" in part:
                                        token = part["text"]
                                        
                                        # Create a chunk in OpenAI format
                                        chunk = {
                                            "id": f"gemini-{model}-{hash(str(original_messages))}",
                                            "object": "chat.completion.chunk",
                                            "created": int(response.elapsed.total_seconds()),
                                            "model": model,
                                            "choices": [
                                                {
                                                    "index": 0,
                                                    "delta": {
                                                        "content": token
                                                    },
                                                    "finish_reason": None
                                                }
                                            ]
                                        }
                                        
                                        yield ChatCompletionChunk(**chunk)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse streaming response line: {line}")
        
        except httpx.HTTPStatusError as e:
            logger.error(f"Google AI API streaming error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error in Google AI chat completion streaming: {str(e)}")
            raise
