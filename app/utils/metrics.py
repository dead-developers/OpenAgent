# app/utils/metrics.py

from collections import defaultdict
import time
from typing import Dict, Any

class MetricsTracker:
    """A simple class to track performance metrics for LLM calls and other operations."""

    def __init__(self):
        self.metrics = defaultdict(lambda: defaultdict(int))
        self.latencies = defaultdict(list)
        self.custom_events = []

    def record_llm_call(self, model_name: str, input_tokens: int, output_tokens: int, latency_ms: float, success: bool = True):
        """Records metrics for an LLM call."""
        self.metrics[model_name]["call_count"] += 1
        self.metrics[model_name]["total_input_tokens"] += input_tokens
        self.metrics[model_name]["total_output_tokens"] += output_tokens
        self.latencies[model_name].append(latency_ms)
        if success:
            self.metrics[model_name]["successful_calls"] += 1
        else:
            self.metrics[model_name]["failed_calls"] += 1

    def record_event(self, event_name: str, category: str = "general", value: Any = 1, **kwargs):
        """Records a custom event or metric."""
        self.metrics[category][event_name] += value if isinstance(value, (int, float)) else 1
        self.custom_events.append({
            "timestamp": time.time(),
            "category": category,
            "event_name": event_name,
            "value": value,
            "details": kwargs
        })

    def get_summary(self) -> Dict[str, Any]:
        """Returns a summary of the collected metrics."""
        summary = {}
        for model_name, data in self.metrics.items():
            summary[model_name] = dict(data)
            if model_name in self.latencies and self.latencies[model_name]:
                avg_latency = sum(self.latencies[model_name]) / len(self.latencies[model_name])
                summary[model_name]["average_latency_ms"] = round(avg_latency, 2)
                summary[model_name]["min_latency_ms"] = round(min(self.latencies[model_name]), 2)
                summary[model_name]["max_latency_ms"] = round(max(self.latencies[model_name]), 2)
        return summary

    def get_custom_events(self) -> list:
        """Returns all recorded custom events."""
        return self.custom_events

    def reset(self):
        """Resets all collected metrics."""
        self.metrics.clear()
        self.latencies.clear()
        self.custom_events.clear()

# Example usage (for testing purposes, would be removed or in a test file)
if __name__ == "__main__":
    tracker = MetricsTracker()
    tracker.record_llm_call("deepseek-v3", 100, 200, 550.5, True)
    tracker.record_llm_call("deepseek-v3", 150, 250, 600.0, True)
    tracker.record_llm_call("deepseek-v3", 120, 0, 300.0, False) # Failed call
    tracker.record_llm_call("deepseek-r1", 80, 150, 400.0, True)

    tracker.record_event("user_login", category="auth", user_id="test_user")
    tracker.record_event("file_processed", category="data_pipeline", file_size_kb=1024)

    print("Summary:", tracker.get_summary())
    print("Custom Events:", tracker.get_custom_events())

