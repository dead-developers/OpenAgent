"""
HuggingFace Hub client integration for OpenAgent using the official huggingface_hub library.
This provides a wrapper that makes the HuggingFace Hub API compatible with OpenAgent's expected format.
"""

import json
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from huggingface_hub import InferenceClient
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage

from app.logger import logger


class HuggingFaceHubClient:
    """Wrapper for the official HuggingFace Hub client to make it compatible with OpenAgent."""

    def __init__(self, api_key: str, model: str = "deepseek-ai/DeepSeek-R1", provider: str = "nebius"):
        self.api_key = api_key
        self.model = model
        self.provider = provider
        self.client = InferenceClient(provider=provider, api_key=api_key)
        self.chat = self.Chat(self)
        logger.info(f"Initialized HuggingFace Hub client for model {self.model} with provider {self.provider}")
    
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
        Create a chat completion using HuggingFace Hub's API via the official Python client.
        
        Args:
            messages: List of conversation messages
            model: Model name (defaults to instance model if not provided)
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream the response
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            ChatCompletion or AsyncIterator[ChatCompletionChunk]
        """
        # Use the model specified in the call, or fall back to the instance model
        model_to_use = model or self.model
        
        try:
            # Set up parameters for the HuggingFace Hub client
            params = {
                "model": model_to_use,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs
            }
            
            if stream:
                # For streaming responses
                return self._stream_chat_completion(params)
            else:
                # For non-streaming responses
                return await self._request_chat_completion(params)
                
        except Exception as e:
            logger.error(f"Error in HuggingFace Hub chat completion: {str(e)}")
            raise
    
    async def _request_chat_completion(self, params: dict) -> ChatCompletion:
        """Make a non-streaming request to HuggingFace Hub API."""
        try:
            # Filter out unsupported parameters
            filtered_params = self._filter_unsupported_params(params)
            
            # Call the HuggingFace Hub API
            completion = self.client.chat.completions.create(**filtered_params)
            
            # Convert to OpenAI format (already compatible)
            return completion
            
        except Exception as e:
            logger.error(f"Error in HuggingFace Hub chat completion: {str(e)}")
            raise
    
    async def _stream_chat_completion(self, params: dict) -> AsyncIterator[ChatCompletionChunk]:
        """Stream a chat completion from HuggingFace Hub API."""
        try:
            # Enable streaming
            params["stream"] = True
            
            # Filter out unsupported parameters
            filtered_params = self._filter_unsupported_params(params)
            
            # Call the HuggingFace Hub API with streaming
            stream = self.client.chat.completions.create(**filtered_params)
            
            # Process the streaming response (already in compatible format)
            for chunk in stream:
                yield chunk
                
        except Exception as e:
            logger.error(f"Error in HuggingFace Hub chat completion streaming: {str(e)}")
            raise
            
    def _filter_unsupported_params(self, params: dict) -> dict:
        """Filter out parameters that are not supported by the HuggingFace Hub API."""
        # List of parameters known to be supported by HuggingFace Hub API
        supported_params = {
            "model", "messages", "temperature", "max_tokens", "stream",
            "top_p", "frequency_penalty", "presence_penalty", "stop"
        }
        
        # Filter out unsupported parameters
        filtered_params = {k: v for k, v in params.items() if k in supported_params}
        
        return filtered_params
