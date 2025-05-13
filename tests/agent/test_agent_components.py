# /home/ubuntu/OpenAgent/tests/agent/test_agent_components.py

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from app.agent.model_router import ModelRouter, DEEPSEEK_V3, DEEPSEEK_R1
from app.agent.dual_model_system import DualModelSystem
from app.agent.supervision import CheckpointSystem, InterventionHandler, FeedbackLoop
from app.config import LLMSettings, Config
from app.schema import Message, ToolChoice
from app.utils.metrics import MetricsTracker

# --- Fixtures for Agent Components ---

@pytest.fixture(scope="module")
def event_loop():
    """Create an instance of the default event loop for each test module."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_llm_configs():
    return {
        DEEPSEEK_V3: LLMSettings(model=DEEPSEEK_V3, api_key="test_key_v3", base_url="test_url_v3", api_type="openai", api_version="v1"),
        DEEPSEEK_R1: LLMSettings(model=DEEPSEEK_R1, api_key="test_key_r1", base_url="test_url_r1", api_type="openai", api_version="v1"),
        "other_model": LLMSettings(model="other/model", api_key="test_key_other", base_url="test_url_other", api_type="openai", api_version="v1")
    }

@pytest.fixture
def model_router(mock_llm_configs):
    return ModelRouter(llm_configs=mock_llm_configs)

@pytest.fixture
def metrics_tracker_instance():
    return MetricsTracker()

@pytest.fixture
def dual_model_system_instance(mock_llm_configs, metrics_tracker_instance):
    # Mock the global config that DualModelSystem tries to read if llm_configs is None
    # This ensures that even if DMS is initialized without explicit configs, it can find them.
    with patch("app.agent.dual_model_system.app_config", MagicMock(spec=Config)) as mock_global_config:
        mock_global_config.llm = mock_llm_configs
        dms = DualModelSystem(llm_configs=mock_llm_configs, metrics_tracker=metrics_tracker_instance)
        return dms

@pytest.fixture
def checkpoint_system_instance():
    return CheckpointSystem()

@pytest.fixture
def mock_agent_or_flow():
    agent_mock = MagicMock()
    agent_mock.apply_intervention = AsyncMock(return_value=True)
    return agent_mock

@pytest.fixture
def intervention_handler_instance(checkpoint_system_instance, mock_agent_or_flow):
    return InterventionHandler(checkpoint_system_instance, mock_agent_or_flow)

@pytest.fixture
def mock_knowledge_store():
    ks_mock = MagicMock()
    ks_mock.store_knowledge = AsyncMock(return_value=None)
    return ks_mock

@pytest.fixture
def feedback_loop_instance(mock_knowledge_store, model_router):
    return FeedbackLoop(knowledge_store=mock_knowledge_store, model_router=model_router)

# --- Tests for ModelRouter ---

def test_model_router_init(model_router, mock_llm_configs):
    assert model_router.llm_configs == mock_llm_configs

def test_model_router_select_model_preference(model_router):
    assert model_router.select_model(preferred_model="deepseek-v3") == DEEPSEEK_V3
    assert model_router.select_model(preferred_model="deepseek-r1") == DEEPSEEK_R1

def test_model_router_select_model_complexity(model_router):
    assert model_router.select_model(task_complexity="high") == DEEPSEEK_V3
    assert model_router.select_model(task_complexity="low") == DEEPSEEK_R1
    assert model_router.select_model(task_complexity="medium") == DEEPSEEK_V3 # Default for medium

def test_model_router_select_model_fallback(model_router, mock_llm_configs):
    # Test fallback if only one DeepSeek model is present
    router_only_r1 = ModelRouter(llm_configs={DEEPSEEK_R1: mock_llm_configs[DEEPSEEK_R1]})
    assert router_only_r1.select_model(task_complexity="high") == DEEPSEEK_R1

    router_no_deepseek = ModelRouter(llm_configs={"other_model": mock_llm_configs["other_model"]})
    assert router_no_deepseek.select_model() == "other_model"

    with pytest.raises(ValueError):
        ModelRouter(llm_configs={}).select_model()

def test_model_router_update_metrics(model_router):
    model_router.update_performance_metrics(DEEPSEEK_V3, {"latency": 100})
    assert model_router.model_performance_metrics[DEEPSEEK_V3] == {"latency": 100}

# --- Tests for DualModelSystem ---

@pytest.mark.asyncio
async def test_dual_model_system_init(dual_model_system_instance, mock_llm_configs):
    assert dual_model_system_instance.router is not None
    assert dual_model_system_instance.metrics_tracker is not None
    # Check if llm_configs in DMS are correctly set up for the router
    assert DEEPSEEK_V3 in dual_model_system_instance.router.llm_configs
    assert DEEPSEEK_R1 in dual_model_system_instance.router.llm_configs

@pytest.mark.asyncio
async def test_dual_model_system_get_llm_instance(dual_model_system_instance):
    # Mock the LLM class constructor to avoid actual API calls or complex setup
    with patch("app.agent.dual_model_system.LLM") as MockLLM:
        mock_llm_instance = MagicMock()
        MockLLM.return_value = mock_llm_instance
        
        llm_v3 = dual_model_system_instance._get_llm_instance(DEEPSEEK_V3)
        assert llm_v3 is mock_llm_instance
        # Ensure LLM was called with a config name that maps to DEEPSEEK_V3 settings
        # This depends on how _get_llm_instance finds the config_key
        # For this test, we assume the mock_llm_configs keys are used as config_name
        MockLLM.assert_called_with(config_name=DEEPSEEK_V3) 

        llm_r1 = dual_model_system_instance._get_llm_instance(DEEPSEEK_R1)
        assert llm_r1 is mock_llm_instance
        MockLLM.assert_called_with(config_name=DEEPSEEK_R1)

@pytest.mark.asyncio
async def test_dual_model_system_execute_llm_call(dual_model_system_instance, metrics_tracker_instance):
    with patch("app.agent.dual_model_system.LLM") as MockLLM:
        mock_llm_instance = AsyncMock() # LLM instance methods are async
        mock_llm_instance.ask = AsyncMock(return_value="Test response")
        mock_llm_instance.ask_tool = AsyncMock(return_value=Message.assistant_message(content="Tool response"))
        mock_llm_instance.count_message_tokens = MagicMock(return_value=10)
        mock_llm_instance.count_tokens = MagicMock(return_value=5)
        mock_llm_instance.format_messages = MagicMock(side_effect=lambda x, **kwargs: x) # Simple pass-through
        mock_llm_instance.MULTIMODAL_MODELS = []
        MockLLM.return_value = mock_llm_instance

        messages = [Message.user_message("Hello")]
        response = await dual_model_system_instance.execute_llm_call(messages=messages, task_complexity="low")
        assert response == "Test response"
        mock_llm_instance.ask.assert_called_once()
        assert metrics_tracker_instance.metrics[DEEPSEEK_R1]["call_count"] == 1
        assert metrics_tracker_instance.metrics[DEEPSEEK_R1]["total_input_tokens"] == 10
        assert metrics_tracker_instance.metrics[DEEPSEEK_R1]["total_output_tokens"] == 5

        # Test tool call
        metrics_tracker_instance.reset() # Reset for clean count
        dummy_tool_spec = {"type": "function", "function": {"name": "dummy"}}
        tool_response = await dual_model_system_instance.execute_llm_call(
            messages=messages, task_complexity="high", tools=[dummy_tool_spec], tool_choice=ToolChoice.AUTO
        )
        assert tool_response.content == "Tool response"
        mock_llm_instance.ask_tool.assert_called_once()
        assert metrics_tracker_instance.metrics[DEEPSEEK_V3]["call_count"] == 1

# --- Tests for CheckpointSystem ---

def test_checkpoint_system_create_get(checkpoint_system_instance):
    ckpt_id = checkpoint_system_instance.create_checkpoint("task1", "step1", "reason1", {"data": "state1"})
    assert ckpt_id is not None
    checkpoint = checkpoint_system_instance.get_checkpoint(ckpt_id)
    assert checkpoint is not None
    assert checkpoint["task_id"] == "task1"
    assert checkpoint["status"] == "pending_review"

def test_checkpoint_system_resolve(checkpoint_system_instance):
    ckpt_id = checkpoint_system_instance.create_checkpoint("task2", "step2", "reason2", {"data": "state2"})
    checkpoint_system_instance.resolve_checkpoint(ckpt_id, {"action": "proceed"}, status="resolved")
    checkpoint = checkpoint_system_instance.get_checkpoint(ckpt_id)
    assert checkpoint["status"] == "resolved"
    assert checkpoint["resolution"] == {"action": "proceed"}

def test_checkpoint_system_get_pending(checkpoint_system_instance):
    checkpoint_system_instance.create_checkpoint("task3", "stepA", "reasonA", {"data": "stateA"})
    checkpoint_system_instance.create_checkpoint("task3", "stepB", "reasonB", {"data": "stateB"})
    ckpt_id_resolved = checkpoint_system_instance.create_checkpoint("task3", "stepC", "reasonC", {"data": "stateC"})
    checkpoint_system_instance.resolve_checkpoint(ckpt_id_resolved, {}, status="resolved")
    
    pending = checkpoint_system_instance.get_pending_checkpoints("task3")
    assert len(pending) == 2
    all_pending = checkpoint_system_instance.get_pending_checkpoints()
    assert len(all_pending) == 2 # Assuming fixture gives clean instance

# --- Tests for InterventionHandler ---

@pytest.mark.asyncio
async def test_intervention_handler_request_intervention(intervention_handler_instance, checkpoint_system_instance):
    ckpt_id = checkpoint_system_instance.create_checkpoint("task_iv", "step_iv", "reason_iv", {})
    assert await intervention_handler_instance.request_intervention(ckpt_id) is True

@pytest.mark.asyncio
async def test_intervention_handler_handle_intervention(intervention_handler_instance, checkpoint_system_instance, mock_agent_or_flow):
    ckpt_id = checkpoint_system_instance.create_checkpoint("task_hv", "step_hv", "reason_hv", {})
    success = await intervention_handler_instance.handle_intervention(ckpt_id, "User feedback", "modify_plan")
    assert success is True
    checkpoint = checkpoint_system_instance.get_checkpoint(ckpt_id)
    assert checkpoint["status"] == "resolved"
    mock_agent_or_flow.apply_intervention.assert_called_once_with(ckpt_id, "User feedback", "modify_plan")

@pytest.mark.asyncio
async def test_intervention_handler_handle_aborted(intervention_handler_instance, checkpoint_system_instance):
    ckpt_id = checkpoint_system_instance.create_checkpoint("task_ab", "step_ab", "reason_ab", {})
    await intervention_handler_instance.handle_intervention(ckpt_id, "Abort task", "abort")
    checkpoint = checkpoint_system_instance.get_checkpoint(ckpt_id)
    assert checkpoint["status"] == "aborted"

# --- Tests for FeedbackLoop ---

@pytest.mark.asyncio
async def test_feedback_loop_record_feedback(feedback_loop_instance):
    feedback_loop_instance.record_feedback("task_fb", "step_fb", "correction", "desc", "user_intervention")
    assert len(feedback_loop_instance.feedback_entries) == 1
    entry = feedback_loop_instance.feedback_entries[0]
    assert entry["task_id"] == "task_fb"
    assert entry["status"] == "pending_processing"

@pytest.mark.asyncio
async def test_feedback_loop_process_feedback(feedback_loop_instance, mock_knowledge_store, model_router):
    feedback_loop_instance.record_feedback(
        "task_p_fb", "step_p_fb", "correction", "desc_corr", "user_intervention",
        data={"corrected_data": [{"id": "d1", "text": "new text"}]}
    )
    feedback_loop_instance.record_feedback(
        "task_p_fb2", None, "negative", "desc_neg", "auto_eval",
        data={"model_id": DEEPSEEK_R1, "model_performance": {"error_increase": 0.1}}
    )
    entry_id_corr = feedback_loop_instance.feedback_entries[0]["feedback_id"]
    entry_id_neg = feedback_loop_instance.feedback_entries[1]["feedback_id"]

    await feedback_loop_instance.process_feedback_entry(entry_id_corr)
    assert feedback_loop_instance.feedback_entries[0]["status"] == "processed"
    mock_knowledge_store.store_knowledge.assert_called_once_with([{"id": "d1", "text": "new text"}])

    await feedback_loop_instance.process_feedback_entry(entry_id_neg)
    assert feedback_loop_instance.feedback_entries[1]["status"] == "processed"
    assert model_router.model_performance_metrics[DEEPSEEK_R1] == {"error_increase": 0.1}

@pytest.mark.asyncio
async def test_feedback_loop_process_all_pending(feedback_loop_instance):
    feedback_loop_instance.record_feedback("t1", "s1", "positive", "d1", "system")
    feedback_loop_instance.record_feedback("t2", "s2", "suggestion", "d2", "user_intervention")
    await feedback_loop_instance.process_all_pending_feedback()
    assert all(fb["status"] == "processed" for fb in feedback_loop_instance.feedback_entries)

# pytest tests/agent/test_agent_components.py

