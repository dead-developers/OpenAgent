import json
import os
import time
import requests
import threading
import tomli
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
from concurrent.futures import ThreadPoolExecutor
from ratelimit import limits, sleep_and_retry
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sentence_transformers import SentenceTransformer
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("deepseek-agent")
Base = declarative_base()

class AgentMemory(Base):
    __tablename__ = 'agent_memory'
    
    id = Column(Integer, primary_key=True)
    subtask = Column(String)
    subtask_embedding = Column(LargeBinary)
    v3_result = Column(String)
    r1_prediction = Column(String)
    differences = Column(String, nullable=True)
    resolution = Column(String, nullable=True)
    rejected_solutions = Column(String, nullable=True)
    subtask_idx = Column(Integer)
    result_embedding = Column(LargeBinary)
    timestamp = Column(DateTime)


class ConfigLoader:
    """Loads configuration from config.toml file and environment variables"""
    
    def __init__(self, config_path: str = None):
        # Use default config path if not provided
        if config_path is None:
            config_path = os.environ.get("MODEL_CONFIG_PATH", "config/config.toml")
        
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> dict:
        """Load the configuration from the TOML file"""
        try:
            with open(self.config_path, "rb") as f:
                config = tomli.load(f)
                logger.info(f"Configuration loaded from {self.config_path}")
                return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise RuntimeError(f"Failed to load configuration from {self.config_path}: {e}")
    
    def _validate_config(self) -> None:
        """Validate that all required configuration is present"""
        # Check primary model configuration
        if "models" not in self.config:
            raise ValueError("Missing 'models' section in configuration")
        
        if "primary" not in self.config["models"]:
            raise ValueError("Missing 'primary' model configuration")
        
        # Check fallback model configuration
        if "fallback" not in self.config["models"]:
            logger.warning("Missing 'fallback' model configuration")
        
        # Check environment variables
        self._resolve_env_vars()
    
    def _resolve_env_vars(self) -> None:
        """Resolve environment variables in the configuration"""
        # Primary model API key
        api_key = self.config["models"]["primary"].get("api_key", "")
        if api_key.startswith("${") and api_key.endswith("}"):
            env_var = api_key[2:-1]
            if env_var not in os.environ:
                logger.warning(f"Environment variable {env_var} not set for primary model API key")
            self.config["models"]["primary"]["api_key"] = os.environ.get(env_var, "")
        
        # Fallback HuggingFace API key
        if "fallback" in self.config["models"]:
            hf_api_key = self.config["models"]["fallback"].get("hf_api_key", "")
            if hf_api_key.startswith("${") and hf_api_key.endswith("}"):
                env_var = hf_api_key[2:-1]
                if env_var not in os.environ:
                    logger.warning(f"Environment variable {env_var} not set for fallback HuggingFace API key")
                self.config["models"]["fallback"]["hf_api_key"] = os.environ.get(env_var, "")
            
            # Fallback OpenRouter API key
            or_api_key = self.config["models"]["fallback"].get("or_api_key", "")
            if or_api_key.startswith("${") and or_api_key.endswith("}"):
                env_var = or_api_key[2:-1]
                if env_var not in os.environ:
                    logger.warning(f"Environment variable {env_var} not set for fallback OpenRouter API key")
                self.config["models"]["fallback"]["or_api_key"] = os.environ.get(env_var, "")
        
        # Vision model API key
        if "vision" in self.config["models"]:
            api_key = self.config["models"]["vision"].get("api_key", "")
            if api_key.startswith("${") and api_key.endswith("}"):
                env_var = api_key[2:-1]
                if env_var not in os.environ:
                    logger.warning(f"Environment variable {env_var} not set for vision model API key")
                self.config["models"]["vision"]["api_key"] = os.environ.get(env_var, "")
    
    def get_config(self) -> dict:
        """Get the full configuration"""
        return self.config
    
    def get_primary_model_config(self) -> dict:
        """Get the primary model configuration"""
        return self.config["models"]["primary"]
    
    def get_fallback_model_config(self) -> dict:
        """Get the fallback model configuration"""
        return self.config["models"]["fallback"] if "fallback" in self.config["models"] else {}
    
    def get_vision_model_config(self) -> dict:
        """Get the vision model configuration"""
        return self.config["models"]["vision"] if "vision" in self.config["models"] else {}
    
    def get_router_config(self) -> dict:
        """Get the router configuration"""
        return self.config.get("router", {"use_smart_routing": True, "max_retries": 3})


class ModelRouter:
    """Handles routing between primary and fallback models with rate limiting and error handling"""
    
    def __init__(self, config: dict):
        self.config = config
        self.router_config = config.get("router", {})
        self.primary_config = config["models"]["primary"]
        self.fallback_config = config["models"].get("fallback", {})
        
        # Rate limiting configuration
        self.max_calls_per_minute = self.router_config.get("max_calls_per_minute", 60)
        self.max_retries = self.router_config.get("max_retries", 3)
        self.retry_delay = self.router_config.get("retry_delay", 2)
        self.timeout = self.router_config.get("timeout_seconds", 30)
        
        # Metrics for smart routing decisions
        self.hf_success_count = 0
        self.hf_failure_count = 0
        self.or_success_count = 0
        self.or_failure_count = 0
        self.last_error_time = 0
    
    @sleep_and_retry
    @limits(calls=60, period=60)
    def call_primary_model(self, prompt: str) -> str:
        """Call the primary model (DeepSeek v3) with rate limiting"""
        model_id = self.primary_config.get("model_id", "deepseek-ai/deepseek-coder-v3-0425-instruct")
        api_key = self.primary_config.get("api_key", "")
        max_tokens = self.primary_config.get("max_tokens", 2048)
        temperature = self.primary_config.get("temperature", 0.1)
        top_p = self.primary_config.get("top_p", 0.95)
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            }
        }
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Calling primary model: {model_id}")
                response = requests.post(
                    f"https://api-inference.huggingface.co/models/{model_id}",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    # Check for malformed response format
                    response_data = response.json()
                    if not isinstance(response_data, list) or len(response_data) == 0 or "generated_text" not in response_data[0]:
                        self.hf_failure_count += 1
                        error_msg = f"Unexpected response format from model: {response_data}"
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                    
                    self.hf_success_count += 1
                    return response_data[0]["generated_text"]
                elif response.status_code == 429:  # Rate limit
                    # Extract retry-after if available
                    retry_after = response.headers.get("retry-after", str(self.retry_delay))
                    try:
                        retry_seconds = int(retry_after)
                    except ValueError:
                        retry_seconds = self.retry_delay
                        
                    self.hf_failure_count += 1
                    error_msg = f"Rate limit exceeded for model API. Please try again in {retry_seconds} seconds."
                    logger.warning(error_msg)
                    last_error = Exception(error_msg)
                    
                    if attempt == self.max_retries - 1:
                        # On last attempt, raise the rate limit exception
                        raise last_error
                    
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    error_msg = f"Primary model error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    self.hf_failure_count += 1
                    self.last_error_time = time.time()
                    last_error = Exception(error_msg)
                    break  # Try fallback for non-rate-limit errors
            except requests.exceptions.Timeout as e:
                error_msg = f"Request to model API timed out: {str(e)}"
                logger.error(error_msg)
                self.hf_failure_count += 1
                self.last_error_time = time.time()
                last_error = Exception(error_msg)
                
                if attempt == self.max_retries - 1:
                    # On last attempt, raise the timeout exception
                    raise last_error
                    
                time.sleep(self.retry_delay)
            except Exception as e:
                logger.error(f"Exception calling primary model: {e}")
                self.hf_failure_count += 1
                self.last_error_time = time.time()
                last_error = e
                time.sleep(self.retry_delay)
        
        # If we get here, all attempts failed
        if last_error:
            raise last_error
        else:
            raise Exception("All attempts to call primary model failed")
    
@sleep_and_retry
    @limits(calls=40, period=60)  # More conservative rate limit for fallback
    def call_fallback_hf_model(self, prompt: str) -> str:
        """Call the HuggingFace fallback model (DeepSeek-R1) with robust error handling.
        
        This implementation mirrors the call_primary_model approach for consistency,
        with specific enhancements for smart routing metrics and fallback scenarios.
        """
        model_id = self.fallback_config.get("hf_model_id", "deepseek-ai/deepseek-ai-r1-instruct")
        api_key = self.fallback_config.get("hf_api_key", "")
        max_tokens = self.fallback_config.get("max_tokens", 2048)
        temperature = self.fallback_config.get("temperature", 0.2)
        top_p = self.fallback_config.get("top_p", 0.9)
        
        # Check if we're currently rate limited
        current_time = time.time()
        if hasattr(self, 'fallback_hf_rate_limited') and self.fallback_hf_rate_limited:
            if hasattr(self, 'fallback_hf_rate_limit_reset') and current_time < self.fallback_hf_rate_limit_reset:
                logger.warning(f"Fallback HF model is rate limited. Will reset in {self.fallback_hf_rate_limit_reset - current_time:.2f} seconds")
                error_msg = f"Rate limit exceeded for fallback HF model API. Try again in {self.fallback_hf_rate_limit_reset - current_time:.2f} seconds."
                raise Exception(error_msg)
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            }
        }
        
        last_error = None
        for attempt in range(self.max_retries):
            if attempt > 0:
                wait_time = self.retry_delay * (2 ** (attempt - 1))
                logger.info(f"Retrying HuggingFace fallback model call in {wait_time:.2f} seconds (attempt {attempt+1}/{self.max_retries})")
                time.sleep(wait_time)
            
            try:
                logger.debug(f"Calling HuggingFace fallback model: {model_id}")
                response = requests.post(
                    f"https://api-inference.huggingface.co/models/{model_id}",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    # Validate the response
                    if not response.text or response.text.strip() == "":
                        error_msg = "HuggingFace fallback model returned empty response"
                        logger.error(error_msg)
                        self.hf_failure_count += 1
                        last_error = ValueError(error_msg)
                        continue  # Try again
                    
                    # Parsing JSON can fail if the response is malformed
                    try:
                        response_data = response.json()
                        if not isinstance(response_data, list) or len(response_data) == 0 or "generated_text" not in response_data[0]:
                            error_msg = f"Unexpected response format from fallback model: {response_data}"
                            logger.error(error_msg)
                            self.hf_failure_count += 1
                            last_error = ValueError(error_msg)
                            continue  # Try again
                        
                        # Success - update metrics for smart routing
                        self.hf_success_count += 1
                        
                        # If we had previously been rate limited, reset that status
                        if hasattr(self, 'fallback_hf_rate_limited') and self.fallback_hf_rate_limited:
                            logger.info("HuggingFace fallback model rate limit has reset")
                            self.fallback_hf_rate_limited = False
                        
                        return response_data[0]["generated_text"]
                    except (json.JSONDecodeError, KeyError, IndexError) as e:
                        error_msg = f"Failed to parse HuggingFace fallback model response: {e}"
                        logger.error(error_msg)
                        self.hf_failure_count += 1
                        last_error = ValueError(error_msg)
                        continue  # Try again
                
                elif response.status_code == 429:  # Rate limit
                    # Extract retry-after if available
                    retry_after = response.headers.get("retry-after", str(self.retry_delay))
                    try:
                        retry_seconds = int(retry_after)
                    except ValueError:
                        retry_seconds = self.retry_delay
                    
                    # Set rate limiting flags for smart routing
                    self.fallback_hf_rate_limited = True
                    self.fallback_hf_rate_limit_reset = current_time + retry_seconds
                    
                    self.hf_failure_count += 1
                    error_msg = f"Rate limit exceeded for fallback model API. Please try again in {retry_seconds} seconds."
                    logger.warning(error_msg)
                    last_error = Exception(error_msg)
                    
                    if attempt == self.max_retries - 1:
                        # On last attempt, raise the rate limit exception
                        raise last_error
                    
                    # Use exponential backoff for retries
                    continue
                
                elif response.status_code >= 500:  # Server errors
                    error_msg = f"HuggingFace fallback server error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    self.hf_failure_count += 1
                    self.last_error_time = current_time
                    last_error = Exception(error_msg)
                    
                    # Server errors might be temporary, so retry
                    continue
                
                else:  # Other errors
                    error_msg = f"HuggingFace fallback model error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    self.hf_failure_count += 1
                    self.last_error_time = current_time
                    last_error = Exception(error_msg)
                    break  # Don't retry for client errors (4xx except 429)
            
            except requests.exceptions.Timeout as e:
                error_msg = f"Request to HuggingFace fallback model API timed out: {str(e)}"
                logger.error(error_msg)
                self.hf_failure_count += 1
                self.last_error_time = current_time
                last_error = Exception(error_msg)
                
                # Timeouts might be temporary, so retry with backoff
                continue
            
            except requests.exceptions.ConnectionError as e:
                error_msg = f"Connection error with HuggingFace fallback model API: {str(e)}"
                logger.error(error_msg)
                self.hf_failure_count += 1
                self.last_error_time = current_time
                last_error = Exception(error_msg)
                
                # Connection errors might be temporary, so retry with backoff
                continue
            
            except Exception as e:
                error_msg = f"Unexpected error calling HuggingFace fallback model: {e}"
                logger.exception(error_msg)  # Log with traceback
                self.hf_failure_count += 1
                self.last_error_time = current_time
                last_error = e
                
                # For unknown errors, retry once but with less confidence
                if attempt >= 1:  # If we've already retried once, break
                    break
        
        # If we get here, all attempts failed
        if last_error:
            logger.error(f"All {self.max_retries} attempts to call HuggingFace fallback model failed")
            raise last_error
        else:
            logger.error(f"All {self.max_retries} attempts to call HuggingFace fallback model failed with unknown errors")
            raise Exception("All attempts to call fallback HuggingFace model failed")
    
    @sleep_and_retry
    @limits(calls=40, period=60)
    def call_fallback_openrouter_model(self, prompt: str) -> str:
        """Call the OpenRouter fallback model (DeepSeek-R1)"""
        model_id = self.fallback_config.get("or_model_id", "deepseek/deepseek-r1:free")
        api_key = self.fallback_config.get("or_api_key", "")
        base_url = self.fallback_config.get("or_base_url", "https://api.openrouter.ai/api/v1")
        max_tokens = self.fallback_config.get("max_tokens", 2048)
        temperature = self.fallback_config.get("temperature", 0.2)
        
        # Get headers from config
        or_headers = self.fallback_config.get("or_headers", {})
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Add custom headers
        for key, value in or_headers.items():
            headers[key] = value
        
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Calling OpenRouter fallback model: {model_id}")
                response = requests.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    # Check for malformed response format
                    response_data = response.json()
                    if not isinstance(response_data, dict) or "choices" not in response_data or \
                       len(response_data["choices"]) == 0 or "message" not in response_data["choices"][0] or \
                       "content" not in response_data["choices"][0]["message"]:
                        self.or_failure_count += 1
                        error_msg = f"Unexpected response format from OpenRouter: {response_data}"
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                    
                    self.or_success_count += 1
                    return response_data["choices"][0]["message"]["content"]
                elif response.status_code == 429:  # Rate limit
                    # Extract retry-after if available
                    retry_after = response.headers.get("retry-after", str(self.retry_delay))
                    try:
                        retry_seconds = int(retry_after)
                    except ValueError:
                        retry_seconds = self.retry_delay
                        
                    self.or_failure_count += 1
                    error_msg = f"Rate limit exceeded for OpenRouter API. Please try again in {retry_seconds} seconds."
                    logger.warning(error_msg)
                    last_error = Exception(error_msg)
                    
                    if attempt == self.max_retries - 1:
                        # On last attempt, raise the rate limit exception
                        raise last_error
                    
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    error_msg = f"OpenRouter fallback model error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    self.or_failure_count += 1
                    self.last_error_time = time.time()
                    last_error = Exception(error_msg)
                    raise last_error
            except requests.exceptions.Timeout as e:
                error_msg = f"Request to OpenRouter API timed out: {str(e)}"
                logger.error(error_msg)
                self.or_failure_count += 1
                self.last_error_time = time.time()
                last_error = Exception(error_msg)
                
                if attempt == self.max_retries - 1:
                    # On last attempt, raise the timeout exception
                    raise last_error
                    
                time.sleep(self.retry_delay)
            except Exception as e:
                logger.error(f"Exception calling OpenRouter fallback model: {e}")
                self.or_failure_count += 1
                self.last_error_time = time.time()
                last_error = e
                time.sleep(self.retry_delay)
        
        # If we get here, all attempts failed
        if last_error:
            raise last_error
        else:
            raise Exception("All attempts to call OpenRouter fallback model failed")
    
    def route_request(self, prompt: str) -> str:
        """Smart route the request to the appropriate model based on current performance metrics"""
        use_smart_routing = self.router_config.get("use_smart_routing", True)
        
        if not use_smart_routing:
            # Always try primary first, then fallbacks in order
            try:
                return self.call_primary_model(prompt)
            except Exception as e:
                logger.warning(f"Primary model failed: {e}")
                try:
                    return self.call_fallback_hf_model(prompt)
                except Exception as e2:
                    logger.warning(f"HuggingFace fallback model failed: {e2}")
                    return self.call_fallback_openrouter_model(prompt)
        
        # Smart routing logic based on recent success rates and errors
        total_primary = max(1, self.hf_success_count + self.hf_failure_count)
        primary_success_rate = self.hf_success_count / total_primary
        
        total_fallback = max(1, self.or_success_count + self.or_failure_count) 
        fallback_success_rate = self.or_success_count / total_fallback
        
        # If primary is doing well (>80% success) or we have no data yet, use primary
        if primary_success_rate > 0.8 or (self.hf_success_count == 0 and self.hf_failure_count == 0):
            try:
                return self.call_primary_model(prompt)
            except Exception as e:
                logger.warning(f"Primary model failed despite good history: {e}")
                # Fall through to fallbacks
        
        # If fallback is doing better than primary, try it first
        if fallback_success_rate > primary_success_rate:
            try:
                # First try HuggingFace fallback
                return self.call_fallback_hf_model(prompt)
            except Exception as e:
                logger.warning(f"HuggingFace fallback model failed: {e}")
                try:
                    # Then try OpenRouter fallback
                    return self.call_fallback_openrouter_model(prompt)
                except Exception as e2:
                    logger.warning(f"All fallback models failed, trying primary as last resort: {e2}")
                    return self.call_primary_model(prompt)
        
        # Default path: try primary, then fallbacks
        try:
            return self.call_primary_model(prompt)
        except Exception as e:
            logger.warning(f"Primary model failed: {e}")
            try:
                return self.call_fallback_hf_model(prompt)
            except Exception as e2:
                logger.warning(f"HuggingFace fallback model failed: {e2}")
                return self.call_fallback_openrouter_model(prompt)
        
    def process_task(self, task_description):
        # Have V3 break down the task into subtasks
        subtasks = self.v3_agent.break_down_task(task_description)
        
        # Begin the execution loop
        final_results = []
        for subtask_idx, subtask in enumerate(subtasks):
            result = self.execute_subtask_loop(subtask, subtask_idx)
            final_results.append(result)
            
        # Synthesize final results if needed
        if len(subtasks) > 1:
            final_result = self.v3_agent.synthesize_results(final_results, task_description)
        else:
            final_result = final_results[0]
            
        return final_result
    
    def execute_subtask_loop(self, subtask, subtask_idx):
        # Start parallel processing
        v3_future = self.thread_pool.submit(self.v3_agent.execute_task, subtask)
        r1_future = self.thread_pool.submit(self.r1_agent.predict_execution, subtask)
        
        # Get results
        v3_result = v3_future.result()
        r1_prediction = r1_future.result()
        
        # Compare results
        differences = self.compare_results(v3_result, r1_prediction)
        
        if not differences:
            # No differences identified - store with no_rejections marker
            self.vector_db.store(
                subtask=subtask,
                v3_result=v3_result,
                r1_prediction=r1_prediction,
                differences=None,
                resolution=None,
                rejected_solutions=".No_rejections",  # Blank file marker as specified
                subtask_idx=subtask_idx
            )
            return v3_result
        
        # Models propose resolutions for differences
        v3_resolution = self.v3_agent.propose_resolution(differences, subtask, v3_result, r1_prediction)
        r1_resolution = self.r1_agent.propose_resolution(differences, subtask, v3_result, r1_prediction)
        
        # Resolve conflicts
        final_resolution = self.resolve_conflicts(
            subtask, v3_resolution, r1_resolution, v3_result, r1_prediction
        )
        
        if final_resolution is None:
            # Both solutions rejected - store failures and restart loop or move to next task
            self.vector_db.store(
                subtask=subtask,
                v3_result=v3_result,
                r1_prediction=r1_prediction,
                differences=differences,
                resolution=None,
                rejected_solutions=[v3_resolution, r1_resolution],
                subtask_idx=subtask_idx
            )
            
            # The requirement is to restart the loop if both are rejected
            # because neither fully completes the task
            return self.execute_subtask_loop(subtask, subtask_idx)
        
        # Store successful execution with no_rejections marker
        self.vector_db.store(
            subtask=subtask,
            v3_result=v3_result,
            r1_prediction=r1_prediction,
            differences=differences,
            resolution=final_resolution,
            rejected_solutions=".No_rejections",  # Blank file marker as specified
            subtask_idx=subtask_idx
        )
        
        return final_resolution
    
    def compare_results(self, v3_result, r1_prediction):
        """Compare the execution result with the prediction to find differences"""
        prompt = f"""
        Compare these two solutions and identify any substantive differences between them:
        
        Solution 1: {v3_result}
        
        Solution 2: {r1_prediction}
        
        Return a JSON array of the key differences, with each difference having:
        - "aspect": The aspect/part that differs
        - "v3_approach": How solution 1 handles it
        - "r1_approach": How solution 2 handles it
        
        If there are no substantive differences, return an empty array.
        """
        
        response = self.r1_agent.raw_call(prompt)
        
        try:
            # Extract JSON from response
            json_str = extract_json(response)
            differences = json.loads(json_str)
            return differences if differences else None
        except:
            # Fallback if JSON parsing fails
            return None
    
    def models_agree(self, resolution1, resolution2):
        """Check if both models agree on all resolutions"""
        # This is a simplified implementation - in practice, you'd need
        # semantic comparison rather than exact string matching
        if isinstance(resolution1, str) and isinstance(resolution2, str):
            # Remove whitespace, newlines, etc. for comparison
            normalized1 = ' '.join(resolution1.split())
            normalized2 = ' '.join(resolution2.split())
            return normalized1 == normalized2
        return False
    
    def evaluate_task_alignment(self, task, solution):
        """Evaluate how closely the solution aligns with the specific task requirements"""
        prompt = f"""
        Evaluate how closely this solution aligns with the specific requirements of the task.
        
        Task: {task}
        
        Solution: {solution}
        
        Return a score from 0.0 to 1.0 where:
        - 0.0 means the solution completely diverges from what was asked
        - 1.0 means the solution perfectly aligns with exactly what was asked for
        
        Just provide the numerical score without explanation.
        """
        
        response = self.r1_agent.raw_call(prompt)
        
        try:
            score = float(response.strip())
            return min(max(score, 0.0), 1.0)
        except:
            return 0.5  # Default to medium alignment
    
    def resolve_conflicts(self, subtask, v3_resolution, r1_resolution, v3_result, r1_prediction):
        # First check if both models agree on all differences
        if self.models_agree(v3_resolution, r1_resolution):
            # No conflict - they agreed on all changes
            return v3_resolution  # Both are the same anyway
            
        # They disagree - each model gets one chance to argue for their choice
        v3_argument = self.v3_agent.present_argument(subtask, v3_resolution, r1_resolution)
        r1_argument = self.r1_agent.present_argument(subtask, v3_resolution, r1_resolution)
        
        # Check if either model concedes based on the other's argument
        v3_agrees_with_r1 = self.v3_agent.evaluate_argument(subtask, r1_argument, v3_resolution, r1_resolution)
        r1_agrees_with_v3 = self.r1_agent.evaluate_argument(subtask, v3_argument, r1_resolution, v3_resolution)
        
        # If one model concedes, use the other's resolution
        if v3_agrees_with_r1:
            return r1_resolution
        elif r1_agrees_with_v3:
            return v3_resolution
            
        # If still disagreement, evaluate based on hard criteria:
        
        # 1. Must fully complete the task
        v3_completion_score = self.evaluate_completion(subtask, v3_resolution)
        r1_completion_score = self.evaluate_completion(subtask, r1_resolution)
        
        # If one fails to complete the task, choose the other
        if v3_completion_score < 1.0 and r1_completion_score >= 1.0:
            return r1_resolution
        elif r1_completion_score < 1.0 and v3_completion_score >= 1.0:
            return v3_resolution
        
        # If both fail to complete the task, reject both
        if v3_completion_score < 1.0 and r1_completion_score < 1.0:
            return None
        
        # 2. Must not add assumed enhancements
        v3_complexity = self.evaluate_complexity(subtask, v3_resolution)
        r1_complexity = self.evaluate_complexity(subtask, r1_resolution)
        
        # If both oversolve, choose the one that oversolves less
        if v3_complexity > 1.0 and r1_complexity > 1.0:
            return v3_resolution if v3_complexity < r1_complexity else r1_resolution
        # If one oversolves, choose the other
        elif v3_complexity > 1.0:
            return r1_resolution
        elif r1_complexity > 1.0:
            return v3_resolution
        
        # 3. Choose the solution that most closely sticks to the original task
        v3_alignment = self.evaluate_task_alignment(subtask, v3_resolution)
        r1_alignment = self.evaluate_task_alignment(subtask, r1_resolution)
        
        return v3_resolution if v3_alignment > r1_alignment else r1_resolution
    
    def evaluate_completion(self, task, solution):
        """Evaluate if the solution completes the task fully (1.0) or partially (<1.0)"""
        prompt = f"""
        Evaluate how completely this solution addresses the given task.
        
        Task: {task}
        
        Solution: {solution}
        
        Return a score from 0.0 to 1.0 where:
        - 0.0 means the solution completely fails to address the task
        - 1.0 means the solution fully addresses all aspects of the task
        
        Just provide the numerical score without explanation.
        """
        
        # Use R1 for evaluation as it's better at reasoning
        response = self.r1_agent.raw_call(prompt)
        
        try:
            score = float(response.strip())
            return min(max(score, 0.0), 1.0)  # Ensure score is between 0 and 1
        except:
            # Default to 0.5 if parsing fails
            return 0.5
    
    def evaluate_complexity(self, task, solution):
        """
        Evaluate solution complexity relative to task requirements.
        Returns:
        - <1.0: Solution is simpler than needed
        - 1.0: Solution matches task complexity
        - >1.0: Solution oversolves/adds unnecessary complexity
        """
        prompt = f"""
        Evaluate whether this solution is unnecessarily complex for the given task.
        
        Task: {task}
        
        Solution: {solution}
        
        Return a score where:
        - 1.0 means the solution has exactly the right complexity for the task
        - <1.0 means the solution is too simple for the task
        - >1.0 means the solution is unnecessarily complex or oversolves the task
        
        Just provide the numerical score without explanation.
        """
        
        response = self.r1_agent.raw_call(prompt)
        
        try:
            score = float(response.strip())
            return max(score, 0.1)  # Ensure minimum score is 0.1
        except:
            return 1.0  # Default to matching complexity
    
    def evaluate_efficiency(self, task, solution):
        """
        Evaluate solution efficiency and elegance.
        Returns a score from 0.0 to 1.0, where higher is better.
        """
        prompt = f"""
        Evaluate how efficient and elegant this solution is for the given task.
        
        Task: {task}
        
        Solution: {solution}
        
        Return a score from 0.0 to 1.0 where:
        - 0.0 means extremely inefficient or inelegant
        - 1.0 means maximally efficient and elegant
        
        Just provide the numerical score without explanation.
        """
        
        response = self.r1_agent.raw_call(prompt)
        
        try:
            score = float(response.strip())
            return min(max(score, 0.0), 1.0)  # Ensure score is between 0 and 1
        except:
            return 0.5  # Default to medium efficiency


class DeepSeekV3Agent:
    def __init__(self, api_key):
        self.api_key = api_key
        self.model_id = "deepseek-ai/deepseek-coder-v3-0425-instruct"
        
    def break_down_task(self, task):
        prompt = f"""
        Break down the following task into a series of smaller, logically sequential subtasks. 
        Each subtask should be clearly defined, manageable, and when completed in sequence, 
        should fulfill the original task.
        
        Task: {task}
        
        Return a JSON array of subtasks, where each subtask is a string.
        """
        
        response = self.call_hf_api(prompt)
        
        # Extract JSON array from response
        try:
            subtasks = json.loads(extract_json(response))
            return subtasks
        except:
            # Fallback to simple text processing if JSON parsing fails
            lines = [line.strip() for line in response.split('\n') 
                    if line.strip() and not line.strip().startswith('```')]
            return [line for line in lines if not line.startswith('#')]
    
    def execute_task(self, task):
        prompt = f"""
        Execute the following task and provide the complete solution:
        
        {task}
        
        Your response should be the direct output needed to complete this task.
        """
        
        return self.call_hf_api(prompt)
    
    def propose_resolution(self, differences, subtask, v3_result, r1_prediction):
        prompt = f"""
        Given the following task and the differences between your solution and the predicted outcome,
        propose your best resolution that addresses these differences.
        
        Task: {subtask}
        
        Your solution: {v3_result}
        
        Predicted outcome: {r1_prediction}
        
        Differences: {json.dumps(differences, indent=2)}
        
        Propose a final solution that resolves these differences. Your proposal should:
        1. Fully complete the original task
        2. Not add unnecessary complexity
        3. Be as efficient and elegant as possible
        
        Provide only the final solution with no explanations.
        """
        
        return self.call_hf_api(prompt)
    
    def present_argument(self, subtask, v3_resolution, r1_resolution):
        """Present logical arguments for why your resolution is better"""
        prompt = f"""
        You need to present a logical argument for why your proposed resolution is better
        than the alternative. Focus on how your solution:
        
        1. Fully completes the original task
        2. Does not add unnecessary complexity 
        3. Most closely aligns with what was specifically asked for
        4. Is more efficient and elegant
        
        Original task: {subtask}
        
        Your resolution: {v3_resolution}
        
        Alternative resolution: {r1_resolution}
        
        Present your logical argument for why your resolution should be chosen:
        """
        
        return self.call_hf_api(prompt)
    
    def evaluate_argument(self, subtask, other_argument, your_resolution, their_resolution):
        """Evaluate if the other model's argument is convincing enough to change your mind"""
        prompt = f"""
        You need to objectively evaluate if the following argument is convincing enough
        for you to change your mind about your proposed resolution.
        
        Original task: {subtask}
        
        Your resolution: {your_resolution}
        
        Their resolution: {their_resolution}
        
        Their argument: {other_argument}
        
        If their argument convincingly demonstrates that their resolution better completes the task,
        is less complex, or more closely aligns with what was specifically asked, you should accept it.
        
        Do you accept their resolution instead of yours? Respond with only YES or NO.
        """
        
        response = self.call_hf_api(prompt)
        return "YES" in response.upper()
    
    def synthesize_results(self, results, original_task):
        prompt = f"""
        Synthesize the following individual results into a cohesive final solution that addresses
        the original task. Ensure the final solution is complete, efficient, and directly addresses
        all aspects of the original task.
        
        Original task: {original_task}
        
        Individual results:
        {json.dumps(results, indent=2)}
        
        Provide the complete synthesized solution.
        """
        
        return self.call_hf_api(prompt)
        
    def call_hf_api(self, prompt):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 2048,
                "temperature": 0.1,
                "top_p": 0.95
            }
        }
        
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{self.model_id}",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()[0]["generated_text"]
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")


class DeepSeekR1Agent:
    def __init__(self, hf_api_key, openrouter_api_key):
        self.hf_api_key = hf_api_key
        self.openrouter_api_key = openrouter_api_key
        self.hf_model_id = "deepseek-ai/deepseek-ai-r1-instruct"
        self.openrouter_model_id = "deepseek/deepseek-r1:free"
        self.use_smart_routing = True
        
    def predict_execution(self, task):
        prompt = f"""
        Given the following task, predict what a successful completion should look like.
        Reason carefully through each step that would be needed to complete the task,
        and describe the expected outcome in detail.
        
        Task: {task}
        
        Your prediction:
        """
        
        return self.smart_call(prompt)
    
    def propose_resolution(self, differences, subtask, v3_result, r1_prediction):
        prompt = f"""
        Given the following task and the differences between the actual solution and your prediction,
        propose your best resolution that addresses these differences.
        
        Task: {subtask}
        
        Actual solution: {v3_result}
        
        Your prediction: {r1_prediction}
        
        Differences: {json.dumps(differences, indent=2)}
        
        Propose a final solution that resolves these differences. Your proposal should:
        1. Fully complete the original task
        2. Not add unnecessary complexity
        3. Be as efficient and elegant as possible
        
        Provide only the final solution with no explanations.
        """
        
        return self.smart_call(prompt)
    
    def present_argument(self, subtask, v3_resolution, r1_resolution):
        """Present logical arguments for why your resolution is better"""
        prompt = f"""
        You need to present a logical argument for why your proposed resolution is better
        than the alternative. Focus on how your solution:
        
        1. Fully completes the original task
        2. Does not add unnecessary complexity 
        3. Most closely aligns with what was specifically asked for
        4. Is more efficient and elegant
        
        Original task: {subtask}
        
        Your resolution: {r1_resolution}
        
        Alternative resolution: {v3_resolution}
        
        Present your logical argument for why your resolution should be chosen:
        """
        
        return self.smart_call(prompt)
    
    def evaluate_argument(self, subtask, other_argument, your_resolution, their_resolution):
        """Evaluate if the other model's argument is convincing enough to change your mind"""
        prompt = f"""
        You need to objectively evaluate if the following argument is convincing enough
        for you to change your mind about your proposed resolution.
        
        Original task: {subtask}
        
        Your resolution: {your_resolution}
        
        Their resolution: {their_resolution}
        
        Their argument: {other_argument}
        
        If their argument convincingly demonstrates that their resolution better completes the task,
        is less complex, or more closely aligns with what was specifically asked, you should accept it.
        
        Do you accept their resolution instead of yours? Respond with only YES or NO.
        """
        
        response = self.smart_call(prompt)
        return "YES" in response.upper()
    
    def raw_call(self, prompt):
        """Direct call to the model for evaluation purposes"""
        return self.smart_call(prompt)
        
    def smart_call(self, prompt):
        """Try HuggingFace first, fall back to OpenRouter if needed"""
        try:
            return self.call_hf_api(prompt)
        except Exception as e:
            print(f"HuggingFace API error: {e}")
            return self.call_openrouter_api(prompt)
            
    def call_hf_api(self, prompt):
        headers = {
            "Authorization": f"Bearer {self.hf_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 2048,
                "temperature": 0.2,
                "top_p": 0.9
            }
        }
        
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{self.hf_model_id}",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()[0]["generated_text"]
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
            
    def call_openrouter_api(self, prompt):
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.openrouter_model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
            "temperature": 0.2,
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"OpenRouter API Error: {response.status_code} - {response.text}")


class VectorDatabase:
    def __init__(self, connection_string="sqlite:///agent_memory.db"):
        self.engine = create_engine(connection_string)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.setup_database()
        
    def setup_database(self):
        Base.metadata.create_all(self.engine)
        
    def store(self, subtask, v3_result, r1_prediction, differences=None, 
              resolution=None, rejected_solutions=None, subtask_idx=0):
        # Create embeddings
        subtask_embedding = self.embedder.encode(subtask)
        result_embedding = self.embedder.encode(str(v3_result))
        
        # Create record
        session = sessionmaker(bind=self.engine)()
        record = AgentMemory(
            subtask=subtask,
            subtask_embedding=subtask_embedding.tobytes(),
            v3_result=json.dumps(v3_result),
            r1_prediction=json.dumps(r1_prediction),
            differences=json.dumps(differences) if differences else None,
            resolution=json.dumps(resolution) if resolution else None,
            rejected_solutions=json.dumps(rejected_solutions) if rejected_solutions else None,
            subtask_idx=subtask_idx,
            result_embedding=result_embedding.tobytes(),
            timestamp=datetime.now()
        )
        
        session.add(record)
        session.commit()
        session.close()
    
    def query_similar(self, query, k=5):
        """Find k most similar tasks based on embeddings"""
        query_embedding = self.embedder.encode(query)
        
        # This is a simplified version - in practice would use a proper vector similarity search
        session = sessionmaker(bind=self.engine)()
        records = session.query(AgentMemory).all()
        session.close()
        
        similarities = []
        for record in records:
            record_embedding = np.frombuffer(record.subtask_embedding, dtype=np.float32)
            similarity = cosine_similarity(query_embedding, record_embedding)
            similarities.append((record, similarity))
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k records
        return [record for record, _ in similarities[:k]]


def extract_json(text):
    """Extract JSON from text that might have markdown code blocks or other content"""
    # Try to extract from code blocks
    import re
    
    # Look for JSON in code blocks
    code_block_pattern = r"```(?:json)?(.*?)```"
    code_blocks = re.findall(code_block_pattern, text, re.DOTALL)
    
    if code_blocks:
        for block in code_blocks:
            try:
                # Validate if it's JSON by trying to parse it
                json.loads(block.strip())
                return block.strip()
            except:
                continue
    
    # Look for arrays/objects directly
    array_pattern = r"\[\s*\".*?\"\s*(?:,\s*\".*?\"\s*)*\]"
    object_pattern = r"\{\s*\".*?\":\s*.*?\s*(?:,\s*\".*?\":\s*.*?\s*)*\}"
    
    array_match = re.search(array_pattern, text, re.DOTALL)
    object_match = re.search(object_pattern, text, re.DOTALL)
    
    if array_match:
        return array_match.group(0)
    elif object_match:
        return object_match.group(0)
    
    # If we can't find JSON, return a simple array with the text lines
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return json.dumps(lines)


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors"""
    dot = sum(a*b for a, b in zip(vec1, vec2))
    norm1 = sum(a*a for a in vec1) ** 0.5
    norm2 = sum(b*b for b in vec2) ** 0.5
    return dot / (norm1 * norm2) if norm1 * norm2 > 0 else 0


class AgentSystem:
    """Main agent system implementing the dual-model architecture with DeepSeek models"""
    
    def __init__(self, config_path=None):
        """Initialize the agent system with configuration"""
        # Load configuration
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.get_config()
        
        # Get primary model API key
        primary_config = self.config_loader.get_primary_model_config()
        hf_api_key = primary_config.get("api_key", "")
        
        # Get fallback model API key
        fallback_config = self.config_loader.get_fallback_model_config()
        or_api_key = fallback_config.get("or_api_key", "")
        
        # Initialize model router
        self.model_router = ModelRouter(self.config)
        
        # Initialize agents
        self.v3_agent = DeepSeekV3Agent(api_key=hf_api_key)
        self.r1_agent = DeepSeekR1Agent(
            hf_api_key=hf_api_key,
            openrouter_api_key=or_api_key
        )
        
        # Initialize vector database
        db_connection = os.environ.get("VECTOR_DB_CONNECTION", "sqlite:///agent_memory.db")
        self.vector_db = VectorDatabase(connection_string=db_connection)
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
    
    def process_task(self, task_description):
        """Process a complete task using the dual-model architecture"""
        # Have V3 break down the task into subtasks
        subtasks = self.v3_agent.break_down_task(task_description)
        
        # Begin the execution loop
        final_results = []
        for subtask_idx, subtask in enumerate(subtasks):
            result = self.execute_subtask_loop(subtask, subtask_idx)
            final_results.append(result)
            
        # Synthesize final results if needed
        if len(subtasks) > 1:
            final_result = self.v3_agent.synthesize_results(final_results, task_description)
        else:
            final_result = final_results[0]
            
        return final_result
    def execute_subtask_loop(self, subtask, subtask_idx):
        """Execute a single subtask using the dual-model loop with verification"""
        # Start parallel processing
        v3_future = self.thread_pool.submit(self.v3_agent.execute_task, subtask)
        r1_future = self.thread_pool.submit(self.r1_agent.predict_execution, subtask)
        
        # Get results
        v3_result = v3_future.result()
        r1_prediction = r1_future.result()
        
        # Compare results
        differences = self.compare_results(v3_result, r1_prediction)
        
        if not differences:
            # No differences identified - store with no_rejections marker
            self.vector_db.store(
                subtask=subtask,
                v3_result=v3_result,
                r1_prediction=r1_prediction,
                differences=None,
                resolution=None,
                rejected_solutions=".No_rejections",  # Blank file marker as specified
                subtask_idx=subtask_idx
            )
            return v3_result
        
        # Models propose resolutions for differences
        v3_resolution = self.v3_agent.propose_resolution(differences, subtask, v3_result, r1_prediction)
        r1_resolution = self.r1_agent.propose_resolution(differences, subtask, v3_result, r1_prediction)
        
        # Resolve conflicts
        final_resolution = self.resolve_conflicts(
            subtask, v3_resolution, r1_resolution, v3_result, r1_prediction
        )
        
        if final_resolution is None:
            # Both solutions rejected - store failures and restart loop or move to next task
            self.vector_db.store(
                subtask=subtask,
                v3_result=v3_result,
                r1_prediction=r1_prediction,
                differences=differences,
                resolution=None,
                rejected_solutions=[v3_resolution, r1_resolution],
                subtask_idx=subtask_idx
            )
            
            # The requirement is to restart the loop if both are rejected
            # because neither fully completes the task
            return self.execute_subtask_loop(subtask, subtask_idx)
        
        # Store successful execution with no_rejections marker
        self.vector_db.store(
            subtask=subtask,
            v3_result=v3_result,
            r1_prediction=r1_prediction,
            differences=differences,
            resolution=final_resolution,
            rejected_solutions=".No_rejections",  # Blank file marker as specified
            subtask_idx=subtask_idx
        )
        
        return final_resolution
    
    def models_agree(self, resolution1, resolution2):
        """Check if both models agree on all resolutions"""
        # This is a simplified implementation - in practice, you'd need
        # semantic comparison rather than exact string matching
        if isinstance(resolution1, str) and isinstance(resolution2, str):
            # Remove whitespace, newlines, etc. for comparison
            normalized1 = ' '.join(resolution1.split())
            normalized2 = ' '.join(resolution2.split())
            return normalized1 == normalized2
        return False
    
    def evaluate_task_alignment(self, task, solution):
        """Evaluate how closely the solution aligns with the specific task requirements"""
        prompt = f"""
        Evaluate how closely this solution aligns with the specific requirements of the task.
        
        Task: {task}
        
        Solution: {solution}
        
        Return a score from 0.0 to 1.0 where:
        - 0.0 means the solution completely diverges from what was asked
        - 1.0 means the solution perfectly aligns with exactly what was asked for
        
        Just provide the numerical score without explanation.
        """
        
        response = self.r1_agent.raw_call(prompt)
        
        try:
            score = float(response.strip())
            return min(max(score, 0.0), 1.0)
        except:
            return 0.5  # Default to medium alignment
    
    def resolve_conflicts(self, subtask, v3_resolution, r1_resolution, v3_result, r1_prediction):
        """
        Resolve conflicts between model solutions using a deterministic set of criteria.
        
        Args:
            subtask: The subtask being solved
            v3_resolution: Resolution proposed by the V3 model
            r1_resolution: Resolution proposed by the R1 model
            v3_result: Original result from V3 model
            r1_prediction: Original prediction from R1 model
            
        Returns:
            The final resolution, or None if both solutions are rejected
        """
        # First check if both models agree on all differences
        if self.models_agree(v3_resolution, r1_resolution):
            # No conflict - they agreed on all changes
            return v3_resolution  # Both are the same anyway
            
        # They disagree - each model gets one chance to argue for their choice
        v3_argument = self.v3_agent.present_argument(subtask, v3_resolution, r1_resolution)
        r1_argument = self.r1_agent.present_argument(subtask, v3_resolution, r1_resolution)
        
        # Check if either model concedes based on the other's argument
        v3_agrees_with_r1 = self.v3_agent.evaluate_argument(subtask, r1_argument, v3_resolution, r1_resolution)
        r1_agrees_with_v3 = self.r1_agent.evaluate_argument(subtask, v3_argument, r1_resolution, v3_resolution)
        
        # If one model concedes, use the other's resolution
        if v3_agrees_with_r1:
            return r1_resolution
        elif r1_agrees_with_v3:
            return v3_resolution
            
        # If still disagreement, evaluate based on hard criteria:
        
        # 1. Must fully complete the task
        v3_completion_score = self.evaluate_completion(subtask, v3_resolution)
        r1_completion_score = self.evaluate_completion(subtask, r1_resolution)
        
        # If one fails to complete the task, choose the other
        if v3_completion_score < 1.0 and r1_completion_score >= 1.0:
            return r1_resolution
        elif r1_completion_score < 1.0 and v3_completion_score >= 1.0:
            return v3_resolution
        
        # If both fail to complete the task, reject both
        if v3_completion_score < 1.0 and r1_completion_score < 1.0:
            return None
        
        # 2. Must not add assumed enhancements
        v3_complexity = self.evaluate_complexity(subtask, v3_resolution)
        r1_complexity = self.evaluate_complexity(subtask, r1_resolution)
        
        # If both oversolve, choose the one that oversolves less
        if v3_complexity > 1.0 and r1_complexity > 1.0:
            return v3_resolution if v3_complexity < r1_complexity else r1_resolution
        # If one oversolves, choose the other
        elif v3_complexity > 1.0:
            return r1_resolution
        elif r1_complexity > 1.0:
            return v3_resolution
        
        # 3. Choose the solution that most closely sticks to the original task
        v3_alignment = self.evaluate_task_alignment(subtask, v3_resolution)
        r1_alignment = self.evaluate_task_alignment(subtask, r1_resolution)
        
        return v3_resolution if v3_alignment > r1_alignment else r1_resolution
    
    def evaluate_completion(self, task, solution):
        """Evaluate if the solution completes the task fully (1.0) or partially (<1.0)"""
        prompt = f"""
        Evaluate how completely this solution addresses the given task.
        
        Task: {task}
        
        Solution: {solution}
        
        Return a score from 0.0 to 1.0 where:
        - 0.0 means the solution completely fails to address the task
        - 1.0 means the solution fully addresses all aspects of the task
        
        Just provide the numerical score without explanation.
        """
        
        # Use R1 for evaluation as it's better at reasoning
        response = self.r1_agent.raw_call(prompt)
        
        try:
            score = float(response.strip())
            return min(max(score, 0.0), 1.0)  # Ensure score is between 0 and 1
        except:
            # Default to 0.5 if parsing fails
            return 0.5
    
    def evaluate_complexity(self, task, solution):
        """
        Evaluate solution complexity relative to task requirements.
        Returns:
        - <1.0: Solution is simpler than needed
        - 1.0: Solution matches task complexity
        - >1.0: Solution oversolves/adds unnecessary complexity
        """
        prompt = f"""
        Evaluate whether this solution is unnecessarily complex for the given task.
        
        Task: {task}
        
        Solution: {solution}
        
        Return a score where:
        - 1.0 means the solution has exactly the right complexity for the task
        - <1.0 means the solution is too simple for the task
        - >1.0 means the solution is unnecessarily complex or oversolves the task
        
        Just provide the numerical score without explanation.
        """
        
        response = self.r1_agent.raw_call(prompt)
        
        try:
            score = float(response.strip())
            return max(score, 0.1)  # Ensure minimum score is 0.1
        except:
            return 1.0  # Default to matching complexity
    
    def compare_results(self, v3_result, r1_prediction):
        """Compare the execution result with the prediction to find differences"""
        prompt = f"""
        Compare these two solutions and identify any substantive differences between them:
        
        Solution 1: {v3_result}
        
        Solution 2: {r1_prediction}
        
        Return a JSON array of the key differences, with each difference having:
        - "aspect": The aspect/part that differs
        - "v3_approach": How solution 1 handles it
        - "r1_approach": How solution 2 handles it
        
        If there are no substantive differences, return an empty array.
        """
        
        response = self.r1_agent.raw_call(prompt)
        
        try:
            # Extract JSON from response
            json_str = extract_json(response)
            differences = json.loads(json_str)
            return differences if differences else None
        except:
            # Fallback if JSON parsing fails
            return None
