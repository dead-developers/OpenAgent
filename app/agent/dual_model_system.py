# app/agent/dual_model_system.py

import time
from typing import Dict, Any, List, Optional, Union, Literal
from app.agent.model_router import ModelRouter, DEEPSEEK_V3, DEEPSEEK_R1
from app.llm import LLM # Existing LLM class
from app.config import LLMSettings, config as app_config # Assuming app_config.llm provides the configs
from app.schema import Message, ToolChoice
from app.logger import logger
from app.utils.metrics import MetricsTracker # Import the metrics tracker

class DualModelSystem:
    """Manages interactions with two DeepSeek models (V3 and R1) via a ModelRouter."""

    def __init__(self, llm_configs: Optional[Dict[str, LLMSettings]] = None, metrics_tracker: Optional[MetricsTracker] = None):
        """
        Initializes the DualModelSystem.

        Args:
            llm_configs (Optional[Dict[str, LLMSettings]]): Specific LLM configurations. 
                                                             If None, loads from global app_config.
            metrics_tracker (Optional[MetricsTracker]): An instance of MetricsTracker.
        """
        if llm_configs is None:
            # Ensure global config is loaded and filter for DeepSeek models if necessary
            # For this system, we are primarily interested in DEEPSEEK_V3 and DEEPSEEK_R1
            # The llm_configs for ModelRouter should ideally contain these.
            # We assume app_config.llm has entries like "deepseek-v3" and "deepseek-r1"
            # which map to their respective LLMSettings.
            self.llm_configs = {}
            if hasattr(app_config, "llm"):
                for model_key, settings in app_config.llm.items():
                    # The model names in config.toml might be like "deepseek_v3_config_key"
                    # We need to map them to the canonical DEEPSEEK_V3/R1 identifiers if router expects that
                    # Or, ensure ModelRouter can work with keys from config.toml
                    # For now, let's assume the keys in app_config.llm are directly usable or mapped
                    # to DEEPSEEK_V3, DEEPSEEK_R1 if they represent these models.
                    # This part might need refinement based on actual config structure.
                    if DEEPSEEK_V3 in settings.model or model_key == "deepseek-v3": # A simple check
                        self.llm_configs[DEEPSEEK_V3] = settings
                    elif DEEPSEEK_R1 in settings.model or model_key == "deepseek-r1":
                        self.llm_configs[DEEPSEEK_R1] = settings
                    else:
                        # Include other models if ModelRouter is designed to handle them too
                        self.llm_configs[model_key] = settings 
            else:
                logger.error("LLM configurations not found in app_config.")
                raise ValueError("LLM configurations are required.")
            
            if not self.llm_configs or (DEEPSEEK_V3 not in self.llm_configs and DEEPSEEK_R1 not in self.llm_configs):
                logger.warning(f"Neither {DEEPSEEK_V3} nor {DEEPSEEK_R1} found in provided llm_configs. Router might not function as expected for these models.")

        else:
            self.llm_configs = llm_configs

        self.router = ModelRouter(llm_configs=self.llm_configs)
        self.llm_instances: Dict[str, LLM] = {}
        self.metrics_tracker = metrics_tracker if metrics_tracker else MetricsTracker() # Use provided or create new

        logger.info("DualModelSystem initialized.")

    def _get_llm_instance(self, model_id: str) -> LLM:
        """Gets or creates an LLM instance for the given model_id."""
        if model_id not in self.llm_instances:
            if model_id in self.llm_configs:
                # The LLM class uses a singleton pattern based on config_name.
                # We need to ensure it can be instantiated with specific model settings if model_id is a full path.
                # For now, assume model_id matches a key in app_config.llm that LLM() can use.
                # Or, we might need to pass LLMSettings directly to LLM constructor if it supports it.
                # The current LLM class takes a `config_name`. If model_id is e.g. "deepseek-ai/DeepSeek-V3",
                # we need a mapping in config.toml like `[llm.deepseek-v3]` where `model = "deepseek-ai/DeepSeek-V3"`
                
                # Find the config key that corresponds to this model_id
                config_key_for_model = None
                for key, settings in self.llm_configs.items():
                    if settings.model == model_id or key == model_id: # Check both actual model name and config key
                        config_key_for_model = key # This key should exist in the global config.llm dictionary
                        break
                
                if not config_key_for_model:
                    # If model_id is a full path like 
                    # "deepseek-ai/DeepSeek-V3", and it's not a direct key in config.llm
                    # we might need to create a temporary LLMSettings or ensure LLM() can handle it.
                    # For now, we assume a direct key or a key whose `model` attribute matches `model_id` exists.
                    logger.warning(f"No direct config key found for model_id 	{model_id}	. Attempting to use it directly if it matches a model name in a config entry.")
                    # This might require LLM() to be more flexible or a more robust mapping here.
                    # A simple approach: if model_id is a full name like 'deepseek-ai/DeepSeek-V3',
                    # and a config entry has `model = "deepseek-ai/DeepSeek-V3"`, use that config entry's key.
                    # This is already handled by the loop above. If no key is found, it's an issue.
                    raise ValueError(f"LLM settings for model {model_id} not found in provided configurations.")

                # Use the found config_key_for_model to instantiate LLM, which expects a config name
                self.llm_instances[model_id] = LLM(config_name=config_key_for_model)
                logger.info(f"LLM instance created for model: {model_id} using config key: {config_key_for_model}")
            else:
                logger.error(f"Configuration for model {model_id} not found.")
                raise ValueError(f"Configuration for model {model_id} not found.")
        return self.llm_instances[model_id]

    async def execute_llm_call(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        task_complexity: Literal["low", "medium", "high"] = "medium",
        preferred_model: Optional[Literal["auto", "deepseek-v3", "deepseek-r1"]] = "auto",
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[ToolChoice] = None,
        stream: bool = False, # Changed default to False as per typical non-streaming use in agents
        temperature: Optional[float] = None,
        **kwargs
    ) -> Any: # Return type can be str for simple ask, or Message object for tool calls
        """
        Selects a model using the router and executes the LLM call.

        Args:
            messages: List of conversation messages.
            system_msgs: Optional system messages.
            task_complexity: Estimated task complexity.
            preferred_model: User's preferred model.
            tools: Optional list of tools for the LLM call.
            tool_choice: Tool choice strategy.
            stream: Whether to stream the response.
            temperature: Sampling temperature.
            **kwargs: Additional parameters for the LLM call.

        Returns:
            The response from the LLM (str or Message with tool_calls).
        """
        selected_model_id = self.router.select_model(task_complexity, preferred_model)
        llm_instance = self._get_llm_instance(selected_model_id)

        logger.info(f"Executing LLM call with model: {selected_model_id}")
        
        start_time = time.time()
        response = None
        success = False
        input_tokens = 0
        output_tokens = 0

        try:
            # Prepare messages for token counting
            # The format_messages is static in LLM, but we need an instance for count_message_tokens
            formatted_messages = llm_instance.format_messages(system_msgs + messages if system_msgs else messages, 
                                                              supports_images=(selected_model_id in llm_instance.MULTIMODAL_MODELS))
            input_tokens = llm_instance.count_message_tokens(formatted_messages)

            if tools:
                response = await llm_instance.ask_tool(
                    messages=messages,
                    system_msgs=system_msgs,
                    tools=tools,
                    tool_choice=tool_choice,
                    temperature=temperature,
                    **kwargs
                )
                # Assuming response is a Message object with tool_calls or content
                if response.content:
                    output_tokens = llm_instance.count_tokens(response.content)
                if response.tool_calls:
                    # This needs a way to count tokens for tool_calls in the response object
                    # For simplicity, let's approximate or assume LLM class handles this internally if it updates its own counters
                    # A more accurate way would be to format the tool_call part of the response and count it.
                    pass # Token counting for tool_call responses might be complex
            else:
                response_str = await llm_instance.ask(
                    messages=messages,
                    system_msgs=system_msgs,
                    stream=stream, # Note: if stream=True, ask returns an AsyncGenerator
                    temperature=temperature,
                    **kwargs
                )
                if stream:
                    # Handle streamed response (this example will just collect it)
                    # In a real scenario, you'd yield chunks
                    collected_response = ""
                    async for chunk in response_str:
                        collected_response += chunk
                    response = collected_response
                    output_tokens = llm_instance.count_tokens(response)
                else:
                    response = response_str
                    output_tokens = llm_instance.count_tokens(response)
            success = True
        except Exception as e:
            logger.error(f"Error during LLM call with {selected_model_id}: {e}")
            # response remains None or could be an error message object
            raise # Re-raise the exception to be handled by the caller
        finally:
            latency_ms = (time.time() - start_time) * 1000
            self.metrics_tracker.record_llm_call(
                model_name=selected_model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                success=success
            )
            # Update router's performance metrics if needed (though router doesn't use them for selection yet)
            # self.router.update_performance_metrics(selected_model_id, self.metrics_tracker.get_summary().get(selected_model_id, {}))

        return response

# Example Usage (for testing purposes)
if __name__ == "__main__":
    async def main():
        # This example assumes you have a config.toml with "deepseek-v3" and "deepseek-r1" sections
        # or that the default LLM config is one of these.
        # Ensure your config.toml has [llm.deepseek-v3] and [llm.deepseek-r1] sections properly defined.
        
        # A mock global config for the example if not running within the full app context
        # This is usually handled by app.config.config
        from app.config import Config
        global_config = Config()
        if not hasattr(global_config, "_config") or not global_config._config.llm.get("deepseek-v3"):
            print("Warning: deepseek-v3 or deepseek-r1 might not be fully configured in your config.toml for this example.")
            print("Please ensure [llm.deepseek-v3] and [llm.deepseek-r1] sections exist with model, api_key etc.")
            # You might need to set up dummy configs for the example to run standalone:
            # global_config._config.llm["deepseek-v3"] = LLMSettings(model=DEEPSEEK_V3, api_key="YOUR_KEY", base_url="YOUR_URL", api_type="openai")
            # global_config._config.llm["deepseek-r1"] = LLMSettings(model=DEEPSEEK_R1, api_key="YOUR_KEY", base_url="YOUR_URL", api_type="openai")

        try:
            dms = DualModelSystem()
            user_message = Message.user_message("Hello, tell me a joke.")
            
            print("--- Testing simple ask (low complexity) ---")
            response_low = await dms.execute_llm_call(
                messages=[user_message],
                task_complexity="low"
            )
            print(f"Response (low complexity): {response_low}")

            print("\n--- Testing simple ask (high complexity) ---")
            response_high = await dms.execute_llm_call(
                messages=[user_message],
                task_complexity="high"
            )
            print(f"Response (high complexity): {response_high}")

            print("\n--- Testing with preferred model (R1) ---")
            response_pref_r1 = await dms.execute_llm_call(
                messages=[user_message],
                preferred_model="deepseek-r1"
            )
            print(f"Response (preferred R1): {response_pref_r1}")

            # Example of a tool (dummy)
            dummy_tool_spec = {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}
                        },
                        "required": ["location"]
                    }
                }
            }
            print("\n--- Testing with tool call ---")
            tool_user_message = Message.user_message("What is the weather like in London?")
            tool_response = await dms.execute_llm_call(
                messages=[tool_user_message],
                tools=[dummy_tool_spec],
                tool_choice=ToolChoice.AUTO,
                task_complexity="medium"
            )
            if isinstance(tool_response, Message) and tool_response.tool_calls:
                print(f"Tool call response: {tool_response.tool_calls}")
            else:
                print(f"Tool call response (or content): {tool_response}")

            print("\n--- Metrics Summary ---")
            print(json.dumps(dms.metrics_tracker.get_summary(), indent=2))

        except ValueError as ve:
            print(f"ValueError during DMS example: {ve}")
        except Exception as e:
            import traceback
            print(f"An unexpected error occurred in DMS example: {e}")
            traceback.print_exc()

    import asyncio
    asyncio.run(main())

