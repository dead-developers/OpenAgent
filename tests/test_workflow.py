#!/usr/bin/env python3
"""
Test script for OpenAgent workflow system validation.

This script tests:
1. Configuration loading
2. Dual-model architecture functionality
3. Vector database operations
4. End-to-end workflow execution
"""

import os
import sys
import unittest
import tempfile
import json
import time
import requests
from typing import Dict, List, Optional, Any
from unittest.mock import patch, MagicMock
from datetime import datetime
# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import workflow modules
from workflow.deepseek_agent_system import (
    ConfigLoader, ModelRouter, VectorDatabase, 
    DeepSeekV3Agent, DeepSeekR1Agent, AgentSystem
)

class TestConfigLoader(unittest.TestCase):
    """Test the configuration loading functionality."""
    
    def setUp(self):
        """Set up test environment variables and temporary config."""
        # Create temporary config file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.temp_dir.name, "test_config.toml")
        
        # Test config content
        config_content = """
        # OpenAgent Dual-Model Architecture Configuration
        
        # Primary Model: DeepSeek v3 0425 from HuggingFace
        [models.primary]
        model_id = "deepseek-ai/deepseek-coder-v3-0425-instruct"
        api_provider = "huggingface"
        api_key = "${HUGGINGFACE_API_KEY}"
        max_tokens = 4096
        temperature = 0.1
        
        # Support Model Configuration: DeepSeek-R1 with smart routing
        [models.fallback]
        hf_model_id = "deepseek-ai/deepseek-ai-r1-instruct"
        hf_api_key = "${HUGGINGFACE_API_KEY}"
        or_model_id = "deepseek/deepseek-r1:free"
        or_base_url = "https://api.openrouter.ai/api/v1"
        or_api_key = "${OPENROUTER_API_KEY}"
        """
        
        with open(self.config_path, "w") as f:
            f.write(config_content)
        
        # Set test environment variables
        os.environ["HUGGINGFACE_API_KEY"] = "test_hf_key_123"
        os.environ["OPENROUTER_API_KEY"] = "test_or_key_456"
    
    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()
        
        # Clean up environment variables
        if "HUGGINGFACE_API_KEY" in os.environ:
            del os.environ["HUGGINGFACE_API_KEY"]
        if "OPENROUTER_API_KEY" in os.environ:
            del os.environ["OPENROUTER_API_KEY"]
    
    def test_config_loading(self):
        """Test that configuration loads correctly."""
        config_loader = ConfigLoader(self.config_path)
        config = config_loader.get_config()
        
        # Verify primary model config
        primary_config = config_loader.get_primary_model_config()
        self.assertEqual(primary_config["model_id"], "deepseek-ai/deepseek-coder-v3-0425-instruct")
        self.assertEqual(primary_config["api_key"], "test_hf_key_123")
        
        # Verify fallback model config
        fallback_config = config_loader.get_fallback_model_config()
        self.assertEqual(fallback_config["hf_model_id"], "deepseek-ai/deepseek-ai-r1-instruct")
        self.assertEqual(fallback_config["hf_api_key"], "test_hf_key_123")
        self.assertEqual(fallback_config["or_api_key"], "test_or_key_456")
    
    def test_missing_env_variables(self):
        """Test handling of missing environment variables."""
        # Remove environment variables
        if "HUGGINGFACE_API_KEY" in os.environ:
            del os.environ["HUGGINGFACE_API_KEY"]
        if "OPENROUTER_API_KEY" in os.environ:
            del os.environ["OPENROUTER_API_KEY"]
        
        config_loader = ConfigLoader(self.config_path)
        config = config_loader.get_config()
        
        # Verify keys are empty strings when env vars are missing
        primary_config = config_loader.get_primary_model_config()
        self.assertEqual(primary_config["api_key"], "")
        
        fallback_config = config_loader.get_fallback_model_config()
        self.assertEqual(fallback_config["hf_api_key"], "")
        self.assertEqual(fallback_config["or_api_key"], "")


class TestModelRouter(unittest.TestCase):
    """Test the model router functionality."""
    
    def setUp(self):
        """Set up mock configuration and router."""
        self.mock_config = {
            "models": {
                "primary": {
                    "model_id": "deepseek-ai/deepseek-coder-v3-0425-instruct",
                    "api_key": "test_hf_key_123"
                },
                "fallback": {
                    "hf_model_id": "deepseek-ai/deepseek-ai-r1-instruct",
                    "hf_api_key": "test_hf_key_123",
                    "or_model_id": "deepseek/deepseek-r1:free",
                    "or_api_key": "test_or_key_456",
                    "or_base_url": "https://api.openrouter.ai/api/v1"
                }
            },
            "router": {
                "use_smart_routing": True,
                "max_retries": 2,
                "timeout_seconds": 10
            }
        }
        
        self.router = ModelRouter(self.mock_config)
    
    @patch('requests.post')
    def test_primary_model_call(self, mock_post: MagicMock):
        """Test primary model call functionality."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"generated_text": "Test response from primary model"}]
        mock_post.return_value = mock_response
        
        result = self.router.call_primary_model("Test prompt")
        self.assertEqual(result, "Test response from primary model")
        self.assertEqual(self.router.hf_success_count, 1)
        self.assertEqual(self.router.hf_failure_count, 0)
    
    @patch('requests.post')
    def test_fallback_model_call(self, mock_post: MagicMock):
        """Test fallback model call functionality."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "Test response from fallback model"}}]}
        mock_post.return_value = mock_response
        
        result = self.router.call_fallback_openrouter_model("Test prompt")
        self.assertEqual(result, "Test response from fallback model")
        self.assertEqual(self.router.or_success_count, 1)
        self.assertEqual(self.router.or_failure_count, 0)
    
    @patch('workflow.deepseek_agent_system.ModelRouter.call_primary_model')
    @patch('workflow.deepseek_agent_system.ModelRouter.call_fallback_hf_model')
    @patch('workflow.deepseek_agent_system.ModelRouter.call_fallback_openrouter_model')
    def test_smart_routing(self, mock_or: MagicMock, mock_hf: MagicMock, mock_primary: MagicMock):
        """Test smart routing logic."""
        # Set up mocks
        mock_primary.return_value = "Primary model response"
        mock_hf.return_value = "HuggingFace fallback response"
        mock_or.return_value = "OpenRouter fallback response"
        
        # First call should try primary model
        result = self.router.route_request("Test prompt")
        self.assertEqual(result, "Primary model response")
        mock_primary.assert_called_once()
        
        # Simulate primary model failure
        mock_primary.side_effect = requests.exceptions.RequestException("Primary model error")
        result = self.router.route_request("Test prompt")
        self.assertEqual(result, "HuggingFace fallback response")
        
        # Simulate HuggingFace fallback failure
        mock_hf.side_effect = requests.exceptions.RequestException("HuggingFace fallback error")
        result = self.router.route_request("Test prompt")
        self.assertEqual(result, "OpenRouter fallback response")

    @patch('requests.post')
    def test_rate_limiting(self, mock_post: MagicMock):
        """Test rate limit detection and backoff."""
        # Create a mock response for rate limiting (HTTP 429)
        rate_limit_response = MagicMock()
        rate_limit_response.status_code = 429
        rate_limit_response.headers = {"retry-after": "30"}  # 30 seconds backoff
        
        # Create a success response for the second call
        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = [{"generated_text": "Successful response after backoff"}]
        
        # First call gets rate limited, second call succeeds
        mock_post.side_effect = [rate_limit_response, success_response]
        
        # First call should detect rate limiting
        with self.assertRaises(Exception) as context:
            self.router.call_primary_model("Test prompt")
        
        # Verify the exception contains information about retry timing
        self.assertIn("Rate limit exceeded", str(context.exception))
        self.assertIn("try again in", str(context.exception))
        
        # Verify rate limiting flag was set
        self.assertTrue(self.router.hf_rate_limited)
        
        # Verify failure count was incremented
        self.assertEqual(self.router.hf_failure_count, 1)
        self.assertEqual(self.router.hf_success_count, 0)
        
        # Mock time.time() to simulate waiting period has passed
        with patch('time.time', return_value=time.time() + 31):
            # Next call should succeed after backoff
            result = self.router.call_primary_model("Test prompt")
            self.assertEqual(result, "Successful response after backoff")
            
            # Verify success count was incremented
            self.assertEqual(self.router.hf_success_count, 1)
    
    @patch('requests.post')
    def test_timeout_handling(self, mock_post: MagicMock):
        """Test API timeout scenarios."""
        # Configure mock to raise a timeout exception
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")
        
        # The call should raise an exception
        with self.assertRaises(Exception) as context:
            self.router.call_primary_model("Test prompt")
        
        # Verify the exception message
        self.assertIn("timed out", str(context.exception))
        
        # Verify failure count was incremented
        self.assertEqual(self.router.hf_failure_count, 1)
        self.assertEqual(self.router.hf_success_count, 0)
        
        # Test with smart routing to see if it falls back to other models
        mock_post.side_effect = [
            requests.exceptions.Timeout("Primary model timeout"),  # Primary model times out
            MagicMock(status_code=200, json=lambda: [{"generated_text": "Fallback model response"}])  # Fallback succeeds
        ]
        
        # Route request should fall back to the working model
        result = self.router.route_request("Test prompt")
        self.assertEqual(result, "Fallback model response")
        
    @patch('requests.post')
    def test_malformed_response(self, mock_post: MagicMock):
        """Test handling of invalid API responses."""
        # Create a mock response with valid status but malformed content
        malformed_response = MagicMock()
        malformed_response.status_code = 200
        malformed_response.json.return_value = {"error": "Something went wrong"}  # Missing expected structure
        
        mock_post.return_value = malformed_response
        
        # The call should raise an exception about unexpected format
        with self.assertRaises(ValueError) as context:
            self.router.call_primary_model("Test prompt")
        
        # Verify the exception message
        self.assertIn("Unexpected response format", str(context.exception))
        
        # Verify failure count was incremented
        self.assertEqual(self.router.hf_failure_count, 1)
        self.assertEqual(self.router.hf_success_count, 0)
        
        # Test with empty list response
        empty_response = MagicMock()
        empty_response.status_code = 200
        empty_response.json.return_value = []  # Empty list
        
        mock_post.return_value = empty_response
        
        # The call should raise an exception
        with self.assertRaises(ValueError) as context:
            self.router.call_primary_model("Test prompt")
        
        # Verify the exception message
        self.assertIn("Unexpected response format", str(context.exception))


class TestVectorDatabase(unittest.TestCase):
    """Test vector database operations."""
    
    def setUp(self):
        """Set up in-memory SQLite database for testing."""
        self.db_connection = "sqlite:///:memory:"
        self.vector_db = VectorDatabase(connection_string=self.db_connection)
    
    def test_store_and_query(self):
        """Test storing data and querying it back."""
        # Test data
        subtask = "Write a function to calculate factorial"
        v3_result = "def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n-1)"
        r1_prediction = "def factorial(n):\n    result = 1\n    for i in range(1, n+1):\n        result *= i\n    return result"
        
        # Store data
        self.vector_db.store(
            subtask=subtask,
            v3_result=v3_result,
            r1_prediction=r1_prediction,
            differences={"approach": "recursion vs iteration"},
            resolution=v3_result,
            rejected_solutions=None,
            subtask_idx=0
        )
        
        # Query similar
        results = self.vector_db.query_similar("Create a factorial function", k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].subtask, subtask)


class TestEndToEndWorkflow(unittest.TestCase):
    """Test end-to-end workflow execution."""
    
    def setUp(self):
        """Set up mocked agent system for testing."""
        # Create patch objects but don't start them yet
        self.config_loader_patch = patch('workflow.deepseek_agent_system.ConfigLoader')
        self.v3_agent_patch = patch('workflow.deepseek_agent_system.DeepSeekV3Agent')
        self.r1_agent_patch = patch('workflow.deepseek_agent_system.DeepSeekR1Agent')
        self.vector_db_patch = patch('workflow.deepseek_agent_system.VectorDatabase')
        
        # Start patches
        self.mock_config_loader = self.config_loader_patch.start()
        self.mock_v3_agent = self.v3_agent_patch.start()
        self.mock_r1_agent = self.r1_agent_patch.start()
        self.mock_vector_db = self.vector_db_patch.start()
        
        # Configure mocks
        mock_config = {
            "models": {
                "primary": {"api_key": "test_hf_key"},
                "fallback": {"hf_api_key": "test_hf_key", "or_api_key": "test_or_key"}
            }
        }
        self.mock_config_loader.return_value.get_config.return_value = mock_config
        self.mock_config_loader.return_value.get_primary_model_config.return_value = mock_config["models"]["primary"]
        self.mock_config_loader.return_value.get_fallback_model_config.return_value = mock_config["models"]["fallback"]
        
        # Set up mock responses
        self.mock_v3_agent.return_value.break_down_task.return_value = ["Subtask 1", "Subtask 2"]
        self.mock_v3_agent.return_value.execute_task.return_value = "V3 result"
        self.mock_r1_agent.return_value.predict_execution.return_value = "R1 prediction"
        
        # Create agent system
        self.agent_system = AgentSystem()
    
    def tearDown(self):
        """Stop all patches."""
        self.config_loader_patch.stop()
        self.v3_agent_patch.stop()
        self.r1_agent_patch.stop()
        self.vector_db_patch.stop()
    
    def test_process_task(self):
        """Test processing a complete task."""
        # Configure mock for compare_results to return no differences
        self.agent_system.compare_results = MagicMock(return_value=None)
        
        result = self.agent_system.process_task("Test task")
        
        # Verify v3_agent.break_down_task was called
        self.mock_v3_agent.return_value.break_down_task.assert_called_once_with("Test task")
        
        # Verify execute_task was called for each subtask
        self.assertEqual(self.mock_v3_agent.return_value.execute_task.call_count, 2)
        
        # Verify predict_execution was called for each subtask
        self.assertEqual(self.mock_r1_agent.return_value.predict_execution.call_count, 2)
        
        # Verify vector_db.store was called for each subtask
        self.assertEqual(self.mock_vector_db.return_value.store.call_count, 2)
        
        # Verify synthesize_results was called
        self.mock_v3_agent.return_value.synthesize_results.assert_called_once()
    
    def test_execute_subtask_with_differences(self):
        """Test executing a subtask with differences."""
        # Configure mock for compare_results to return differences
        self.agent_system.compare_results = MagicMock(return_value=["difference1"])
        
        # Configure resolution proposals
        self.mock_v3_agent.return_value.propose_resolution.return_value = "V3 resolution"
        self.mock_r1_agent.return_value.propose_resolution.return_value = "R1 resolution"
        
        # Configure models_agree to return True
        self.agent_system.models_agree = MagicMock(return_value=True)
        
        result = self.agent_system.execute_subtask_loop("Test subtask", 0)
        
        # Verify both agents were called to execute/predict
        self.mock_v3_agent.return_value.execute_task.assert_called_once_with("Test subtask")
        self.mock_r1_agent.return_value.predict_execution.assert_called_once_with("Test subtask")
        
        # Verify comparison was done
        self.agent_system.compare_results.assert_called_once()
        
        # Verify resolutions were proposed
        self.mock_v3_agent.return_value.propose_resolution.assert_called_once()
        self.mock_r1_agent.return_value.propose_resolution.assert_called_once()

