# app/agent/model_router.py

from typing import Dict, Any, Optional, Literal
from app.logger import logger
from app.config import LLMSettings # Assuming LLMSettings can be used or adapted

# Define model identifiers as per specification
DEEPSEEK_V3 = "deepseek-ai/DeepSeek-V3"
DEEPSEEK_R1 = "deepseek-ai/DeepSeek-R1"

class ModelRouter:
    """Routes requests to appropriate LLMs based on criteria like complexity, performance, and resources."""

    def __init__(self, llm_configs: Dict[str, LLMSettings]):
        """
        Initializes the ModelRouter with available LLM configurations.

        Args:
            llm_configs (Dict[str, LLMSettings]): A dictionary where keys are model identifiers
                                                 (e.g., "deepseek-v3", "deepseek-r1") and
                                                 values are LLMSettings objects.
        """
        self.llm_configs = llm_configs
        self.model_performance_metrics: Dict[str, Dict[str, Any]] = {}
        logger.info(f"ModelRouter initialized with models: {list(llm_configs.keys())}")

    def select_model(
        self,
        task_complexity: Literal["low", "medium", "high"] = "medium",
        preferred_model: Optional[Literal["auto", "deepseek-v3", "deepseek-r1"]] = "auto",
        current_load: Optional[Dict[str, float]] = None # e.g., {"deepseek-v3_gpu_util": 0.8}
    ) -> str:
        """
        Selects the most appropriate model for a given task.

        Args:
            task_complexity (str): Estimated complexity of the task ("low", "medium", "high").
            preferred_model (str): User or system preference ("auto", "deepseek-v3", "deepseek-r1").
            current_load (Optional[Dict[str, float]]): Current resource load on available models/systems.

        Returns:
            str: The identifier of the selected model (e.g., DEEPSEEK_V3 or DEEPSEEK_R1).
        """
        logger.debug(f"Selecting model with complexity: {task_complexity}, preference: {preferred_model}")

        # Direct preference if not "auto"
        if preferred_model == "deepseek-v3" and DEEPSEEK_V3 in self.llm_configs:
            logger.info(f"Using preferred model: {DEEPSEEK_V3}")
            return DEEPSEEK_V3
        if preferred_model == "deepseek-r1" and DEEPSEEK_R1 in self.llm_configs:
            logger.info(f"Using preferred model: {DEEPSEEK_R1}")
            return DEEPSEEK_R1

        # Default to a capable model if available (e.g., V3 for high complexity)
        # This logic can be significantly expanded based on actual model capabilities and metrics
        # For now, a simple heuristic:
        if task_complexity == "high" and DEEPSEEK_V3 in self.llm_configs:
            logger.info(f"High complexity task, selecting: {DEEPSEEK_V3}")
            return DEEPSEEK_V3
        elif task_complexity == "low" and DEEPSEEK_R1 in self.llm_configs:
            logger.info(f"Low complexity task, selecting: {DEEPSEEK_R1}")
            return DEEPSEEK_R1
        
        # Fallback logic: prefer V3 if available, then R1
        if DEEPSEEK_V3 in self.llm_configs:
            logger.info(f"Defaulting to capable model: {DEEPSEEK_V3}")
            return DEEPSEEK_V3
        if DEEPSEEK_R1 in self.llm_configs:
            logger.info(f"Defaulting to available model: {DEEPSEEK_R1}")
            return DEEPSEEK_R1

        # If no specific DeepSeek models are configured, raise error or return a default from llm_configs
        if self.llm_configs:
            fallback_model = list(self.llm_configs.keys())[0]
            logger.warning(f"No DeepSeek models available or matched, falling back to first configured model: {fallback_model}")
            return fallback_model
        
        logger.error("No models available in ModelRouter configuration.")
        raise ValueError("No models configured for ModelRouter.")

    def update_performance_metrics(self, model_id: str, metrics: Dict[str, Any]):
        """
        Updates the performance metrics for a given model.
        This would be called by the MetricsTracker or a similar mechanism.

        Args:
            model_id (str): The identifier of the model.
            metrics (Dict[str, Any]): A dictionary of performance metrics.
                                      (e.g., {"avg_latency_ms": 500, "error_rate": 0.05})
        """
        self.model_performance_metrics[model_id] = metrics
        logger.info(f"Updated performance metrics for {model_id}: {metrics}")

# Example Usage (for testing purposes)
if __name__ == "__main__":
    # Dummy LLMSettings for testing
    class DummyLLMSettings:
        def __init__(self, model_name):
            self.model = model_name
            # Add other fields as expected by LLMSettings if necessary for router logic

    configs = {
        DEEPSEEK_V3: DummyLLMSettings(DEEPSEEK_V3),
        DEEPSEEK_R1: DummyLLMSettings(DEEPSEEK_R1)
    }
    router = ModelRouter(llm_configs=configs)

    print(f"Selected for low complexity, auto: {router.select_model(task_complexity=	low	)}")
    print(f"Selected for high complexity, auto: {router.select_model(task_complexity=	high	)}")
    print(f"Selected for medium complexity, preferred r1: {router.select_model(preferred_model=	deepseek-r1	)}")
    print(f"Selected for medium complexity, preferred v3: {router.select_model(preferred_model=	deepseek-v3	)}")

    router.update_performance_metrics(DEEPSEEK_V3, {"avg_latency_ms": 300, "cost_per_1k_tokens": 0.02})
    print(f"Metrics for {DEEPSEEK_V3}: {router.model_performance_metrics.get(DEEPSEEK_V3)}")

