import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List

import pytest
from pydantic import ValidationError

from app.models.base import (
    MemoryType, ExecutionStatus, MemoryRetentionPolicy, ActionType, 
    ThoughtType, ModelProvider, MemoryEntry, AgentAction, AgentThought,
    ActionParameter, ModelConfig, MemoryConfig, AgentConfig
)


# ==================== FIXTURES ====================

@pytest.fixture
def valid_memory_entry() -> MemoryEntry:
    """Return a valid memory entry for testing."""
    return MemoryEntry(
        type=MemoryType.CONVERSATION,
        content={"role": "user", "message": "Hello, world!"},
        tags=["greeting", "test"]
    )


@pytest.fixture
def valid_action_parameter() -> ActionParameter:
    """Return a valid action parameter for testing."""
    return ActionParameter(
        name="query",
        value="test query",
        description="Search query parameter"
    )


@pytest.fixture
def valid_agent_action(valid_action_parameter) -> AgentAction:
    """Return a valid agent action for testing."""
    return AgentAction(
        type=ActionType.TOOL_CALL,
        name="search",
        description="Search for information",
        parameters=[valid_action_parameter]
    )


@pytest.fixture
def valid_agent_thought() -> AgentThought:
    """Return a valid agent thought for testing."""
    return AgentThought(
        type=ThoughtType.REASONING,
        content="I should search for information about this topic.",
        supporting_facts=["The user asked about this topic.", "I don't have enough information."]
    )


@pytest.fixture
def valid_model_config() -> ModelConfig:
    """Return a valid model configuration for testing."""
    return ModelConfig(
        model_name="gpt-4",
        provider=ModelProvider.OPENAI,
        api_key="sk-test123"
    )


@pytest.fixture
def valid_memory_config() -> MemoryConfig:
    """Return a valid memory configuration for testing."""
    return MemoryConfig(
        max_entries=5000,
        use_embeddings=True,
        embedding_model="text-embedding-ada-002"
    )


@pytest.fixture
def valid_agent_config(valid_model_config) -> AgentConfig:
    """Return a valid agent configuration for testing."""
    return AgentConfig(
        name="TestAgent",
        description="A test agent",
        model=valid_model_config,
        tools=["search", "calculator"]
    )


# ==================== TESTS FOR ENUMS ====================

def test_memory_type_enum():
    """Test MemoryType enum values."""
    assert MemoryType.CONVERSATION == "conversation"
    assert MemoryType.ACTION == "action"
    assert MemoryType.THOUGHT == "thought"
    assert MemoryType.OBSERVATION == "observation"
    assert MemoryType.PLAN == "plan"
    assert MemoryType.METADATA == "metadata"
    assert MemoryType.EXTERNAL == "external"


def test_execution_status_enum():
    """Test ExecutionStatus enum values."""
    assert ExecutionStatus.PENDING == "pending"
    assert ExecutionStatus.IN_PROGRESS == "in_progress"
    assert ExecutionStatus.COMPLETED == "completed"
    assert ExecutionStatus.FAILED == "failed"
    assert ExecutionStatus.CANCELLED == "cancelled"
    assert ExecutionStatus.TIMEOUT == "timeout"


def test_memory_retention_policy_enum():
    """Test MemoryRetentionPolicy enum values."""
    assert MemoryRetentionPolicy.PERMANENT == "permanent"
    assert MemoryRetentionPolicy.SESSION == "session"
    assert MemoryRetentionPolicy.TEMPORARY == "temporary"
    assert MemoryRetentionPolicy.EXPIRING == "expiring"


def test_action_type_enum():
    """Test ActionType enum values."""
    assert ActionType.TOOL_CALL == "tool_call"
    assert ActionType.API_CALL == "api_call"
    assert ActionType.INTERNAL == "internal"
    assert ActionType.MESSAGE == "message"
    assert ActionType.QUERY == "query"
    assert ActionType.DECISION == "decision"
    assert ActionType.PLANNING == "planning"


def test_thought_type_enum():
    """Test ThoughtType enum values."""
    assert ThoughtType.REASONING == "reasoning"
    assert ThoughtType.PLANNING == "planning"
    assert ThoughtType.EVALUATION == "evaluation"
    assert ThoughtType.REFLECTION == "reflection"
    assert ThoughtType.CREATIVITY == "creativity"
    assert ThoughtType.SUMMARY == "summary"
    assert ThoughtType.QUESTION == "question"


def test_model_provider_enum():
    """Test ModelProvider enum values."""
    assert ModelProvider.OPENAI == "openai"
    assert ModelProvider.ANTHROPIC == "anthropic"
    assert ModelProvider.AZURE == "azure"
    assert ModelProvider.GOOGLE == "google"
    assert ModelProvider.OPEN_SOURCE == "open_source"
    assert ModelProvider.OPENROUTER == "openrouter"
    assert ModelProvider.FIREWORKS == "fireworks"
    assert ModelProvider.LOCAL == "local"
    assert ModelProvider.CUSTOM == "custom"


# ==================== TESTS FOR MEMORY ENTRY ====================

def test_memory_entry_basic_instantiation(valid_memory_entry):
    """Test that a memory entry can be instantiated with valid data."""
    assert valid_memory_entry.type == MemoryType.CONVERSATION
    assert valid_memory_entry.content == {"role": "user", "message": "Hello, world!"}
    assert "greeting" in valid_memory_entry.tags
    assert "test" in valid_memory_entry.tags
    assert valid_memory_entry.retention == MemoryRetentionPolicy.SESSION
    assert valid_memory_entry.importance == 0.5
    assert isinstance(valid_memory_entry.id, uuid.UUID)
    assert isinstance(valid_memory_entry.timestamp, datetime)


def test_memory_entry_conversation_validation():
    """Test that conversation memory entries are validated correctly."""
    # Valid conversation
    valid_entry = MemoryEntry(
        type=MemoryType.CONVERSATION,
        content={"role": "assistant", "message": "How can I help?"}
    )
    assert valid_entry is not None

    # Valid conversation with list
    valid_list_entry = MemoryEntry(
        type=MemoryType.CONVERSATION,
        content=[
            {"role": "user", "message": "Hello"},
            {"role": "assistant", "message": "Hi there"}
        ]
    )
    assert valid_list_entry is not None

    # Invalid conversation (not a dict or list)
    with pytest.raises(ValidationError):
        MemoryEntry(type=MemoryType.CONVERSATION, content="Just a string")


def test_memory_entry_expiring():
    """Test expiring memory entries."""
    # Valid expiring memory
    future_time = datetime.utcnow() + timedelta(days=1)
    valid_expiring = MemoryEntry(
        type=MemoryType.METADATA,
        content={"key": "value"},
        retention=MemoryRetentionPolicy.EXPIRING,
        expiration=future_time
    )
    assert valid_expiring is not None

    # Invalid expiring memory (missing expiration)
    with pytest.raises(ValidationError):
        MemoryEntry(
            type=MemoryType.METADATA,
            content={"key": "value"},
            retention=MemoryRetentionPolicy.EXPIRING
        )


def test_memory_entry_embedding():
    """Test memory entries with embeddings."""
    # Valid embedding
    valid_embedding = MemoryEntry(
        type=MemoryType.EXTERNAL,
        content={"source": "wikipedia", "text": "Information"},
        embedding=[0.1, 0.2, 0.3, 0.4]
    )
    assert valid_embedding is not None

    # Invalid embedding (not all floats)
    with pytest.raises(ValidationError):
        MemoryEntry(
            type=MemoryType.EXTERNAL,
            content={"source": "wikipedia", "text": "Information"},
            embedding=[0.1, "not a float", 0.3]
        )


def test_memory_entry_to_dict(valid_memory_entry):
    """Test conversion to dictionary."""
    memory_dict = valid_memory_entry.to_dict()
    
    assert isinstance(memory_dict, dict)
    assert memory_dict["id"] == str(valid_memory_entry.id)
    assert memory_dict["type"] == "conversation"
    assert memory_dict["content"] == {"role": "user", "message": "Hello, world!"}
    assert memory_dict["tags"] == ["greeting", "test"]
    assert memory_dict["retention"] == "session"
    assert memory_dict["importance"] == 0.5
    assert memory_dict["session_id"] is None
    assert memory_dict["related_entries"] == []
    assert memory_dict["expiration"] is None


def test_memory_entry_edge_cases():
    """Test edge cases for memory entries."""
    # Test with minimum importance
    min_importance = MemoryEntry(
        type=MemoryType.METADATA,
        content={"key": "value"},
        importance=0.0
    )
    assert min_importance.importance == 0.0

    # Test with maximum importance
    max_importance = MemoryEntry(
        type=MemoryType.METADATA,
        content={"key": "value"},
        importance=1.0
    )
    assert max_importance.importance == 1.0
    
    # Test with importance out of range
    with pytest.raises(ValidationError):
        MemoryEntry(
            type=MemoryType.METADATA,
            content={"key": "value"},
            importance=1.1
        )


# ==================== TESTS FOR AGENT ACTION ====================

def test_agent_action_basic_instantiation(valid_agent_action):
    """Test that an agent action can be instantiated with valid data."""
    assert valid_agent_action.type == ActionType.TOOL_CALL
    assert valid_agent_action.name == "search"
    assert valid_agent_action.description == "Search for information"
    assert len(valid_agent_action.parameters) == 1
    assert valid_agent_action.parameters[0].name == "query"
    assert valid_agent_action.status == ExecutionStatus.PENDING
    assert valid_agent_action.error is None
    assert valid_agent_action.result is None
    assert isinstance(valid_agent_action.id, uuid.UUID)
    assert isinstance(valid_agent_action.timestamp, datetime)


def test_agent_action_completed():
    """Test completed agent actions."""
    # Create a completed action
    action = AgentAction(
        type=ActionType.TOOL_CALL,
        name="search",
        status=ExecutionStatus.COMPLETED,
        result={"found": True, "data": "search results"}
    )
    assert action.status == ExecutionStatus.COMPLETED
    assert action.result == {"found": True, "data": "search results"}

    # Should fail if status is COMPLETED but result is None
    with pytest.raises(ValidationError):
        AgentAction(
            type=ActionType.TOOL_CALL,
            name="search",
            status=ExecutionStatus.COMPLETED
        )


def test_agent_action_failed():
    """Test failed agent actions."""
    # Create a failed action
    action = AgentAction(
        type=ActionType.TOOL_CALL,
        name="search",
        status=ExecutionStatus.FAILED,
        error="Search API is down"
    )
    assert action.status == ExecutionStatus.FAILED
    assert action.error == "Search API is down"

    # Should fail if status is FAILED but error is None
    with pytest.raises(ValidationError):
        AgentAction(
            type=ActionType.TOOL_CALL,
            name="search",
            status=ExecutionStatus.FAILED
        )


def test_agent_action_to_dict(valid_agent_action):
    """Test conversion to dictionary."""
    action_dict = valid_agent_action.to_dict()
    
    assert isinstance(action_dict, dict)
    assert action_dict["id"] == str(valid_agent_action.id)
    assert action_dict["type"] == "tool_call"
    assert action_dict["name"] == "search"
    assert action_dict["description"] == "Search for information"
    assert action_dict["status"] == "pending"
    assert action_dict["error"] is None
    assert action_dict["result"] is None
    assert len(action_dict["parameters"]) == 1
    assert action_dict["parameters"][0]["name"] == "query"


def test_agent_action_with_methods():
    """Test action helper methods."""
    # Create a base action
    action = AgentAction(
        type=ActionType.API_CALL,
        name="weather",
        parameters=[ActionParameter(name="location", value="New York", type_hint="string")]
    )
    
    # Test with_result
    completed_action = action.with_result(
        result={"temperature": 72, "condition": "sunny"},
        execution_time=0.35
    )
    assert completed_action.status == ExecutionStatus.COMPLETED
    assert completed_action.result == {"temperature": 72, "condition": "sunny"}
    assert completed_action.execution_time == 0.35
    
    # Test with_error
    failed_action = action.with_error(
        error="API rate limit exceeded",
        execution_time=0.1
    )
    assert failed_action.status == ExecutionStatus.FAILED
    assert failed_action.error == "API rate limit exceeded"
    assert failed_action.execution_time == 0.1


# ==================== TESTS FOR AGENT THOUGHT ====================

def test_agent_thought_basic_instantiation(valid_agent_thought):
    """Test that an agent thought can be instantiated with valid data."""
    assert valid_agent_thought.type == ThoughtType.REASONING
    assert valid_agent_thought.content == "I should search for information about this topic."
    assert len(valid_agent_thought.supporting_facts) == 2
    assert valid_agent_thought.confidence == 0.5
    assert isinstance(valid_agent_thought.id, uuid.UUID)
    assert isinstance(valid_agent_thought.timestamp, datetime)


def test_agent_thought_content_validation():
    """Test thought content validation."""
    # Valid thought
    valid_thought = AgentThought(
        type=ThoughtType.EVALUATION,
        content="This is a valid thought"
    )
    assert valid_thought is not None

    # Empty content
    with pytest.raises(ValidationError):
        AgentThought(
            type=ThoughtType.EVALUATION,
            content=""
        )

    # Whitespace-only content
    with pytest.raises(ValidationError):
        AgentThought(
            type=ThoughtType.EVALUATION,
            content="   "
        )


def test_agent_thought_confidence():
    """Test thought confidence validation."""
    # Minimum confidence
    min_confidence = AgentThought(
        type=ThoughtType.REASONING,
        content="Low confidence thought",
        confidence=0.0
    )
    assert min_confidence.confidence == 0.0

    # Maximum confidence
    max_confidence = AgentThought(
        type=ThoughtType.REASONING,
        content="High confidence thought",
        confidence=1.0
    )
    assert max_confidence.confidence == 1.0

    # Invalid confidence (too low)
    with pytest.raises(ValidationError):
        AgentThought(
            type=ThoughtType.REASONING,
            content="Invalid confidence",
            confidence=-0.1
        )

    # Invalid confidence (too high)
    with pytest.raises(ValidationError):
        AgentThought(
            type=ThoughtType.REASONING,
            content="Invalid confidence",
            confidence=1.1
        )


def test_agent_thought_to_dict(valid_agent_thought):
    """Test conversion to dictionary."""
    thought_dict = valid_agent_thought.to_dict()
    
    assert isinstance(thought_dict, dict)
    assert thought_dict["id"] == str(valid_agent_thought.id)
    assert thought_dict["type"] == "reasoning"
    assert thought_dict["content"] == "I should search for information about this topic."
    assert thought_dict["supporting_facts"] == ["The user asked about this topic.", "I don't have enough information."]
    assert thought_dict["confidence"] == 0.5


def test_agent_thought_supporting_facts():
    """Test supporting facts validation."""
    # Empty supporting facts list
    empty_facts = AgentThought(
        type=ThoughtType.REASONING,
        content="A thought with no supporting facts",
        supporting_facts=[]
    )
    assert empty_facts.supporting_facts == []

    # Valid supporting facts
    valid_facts = AgentThought(
        type=ThoughtType.REASONING,
        content="A thought with supporting facts",
        supporting_facts=["Fact 1", "Fact 2", "Fact 3"]
    )
    assert len(valid_facts.supporting_facts) == 3

    # Invalid supporting facts (non-string item)
    with pytest.raises(ValidationError):
        AgentThought(
            type=ThoughtType.REASONING,
            content="Invalid supporting facts",
            supporting_facts=["Valid fact", 123, "Another valid fact"]
        )


# ==================== TESTS FOR MODEL CONFIG ====================

def test_model_config_basic_instantiation(valid_model_config):
    """Test that a model config can be instantiated with valid data."""
    assert valid_model_config.model_name == "gpt-4"
    assert valid_model_config.provider == ModelProvider.OPENAI
    assert valid_model_config.api_key == "sk-test123"
    assert valid_model_config.temperature == 0.7  # Default value
    assert valid_model_config.max_tokens is None  # Default value


def test_model_config_parameters():
    """Test model config with parameters."""
    config = ModelConfig(
        model_name="gpt-4-turbo",
        provider=ModelProvider.OPENAI,
        api_key="sk-test456",
        temperature=0.4,
        max_tokens=2000,
        top_p=0.95,
        presence_penalty=0.2,
        frequency_penalty=0.3,
        system_prompt="You are a helpful AI assistant."
    )
    
    assert config.model_name == "gpt-4-turbo"
    assert config.temperature == 0.4
    assert config.max_tokens == 2000
    assert config.top_p == 0.95
    assert config.presence_penalty == 0.2
    assert config.frequency_penalty == 0.3
    assert config.system_prompt == "You are a helpful AI assistant."


def test_model_config_temperature_validation():
    """Test temperature validation in model config."""
    # Minimum temperature
    min_temp = ModelConfig(
        model_name="gpt-4",
        provider=ModelProvider.OPENAI,
        temperature=0.0
    )
    assert min_temp.temperature == 0.0
    
    # Maximum temperature
    max_temp = ModelConfig(
        model_name="gpt-4",
        provider=ModelProvider.OPENAI,
        temperature=1.0
    )
    assert max_temp.temperature == 1.0
    
    # Invalid temperature (too low)
    with pytest.raises(ValidationError):
        ModelConfig(
            model_name="gpt-4",
            provider=ModelProvider.OPENAI,
            temperature=-0.1
        )
        
    # Invalid temperature (too high)
    with pytest.raises(ValidationError):
        ModelConfig(
            model_name="gpt-4",
            provider=ModelProvider.OPENAI,
            temperature=1.1
        )


def test_model_config_to_dict(valid_model_config):
    """Test conversion to dictionary."""
    config_dict = valid_model_config.to_dict()
    
    assert isinstance(config_dict, dict)
    assert config_dict["model_name"] == "gpt-4"
    assert config_dict["provider"] == "openai"
    assert config_dict["api_key"] == "sk-test123"
    assert config_dict["temperature"] == 0.7
    assert config_dict["max_tokens"] is None


def test_model_config_no_api_key():
    """Test model config without API key."""
    # Should work for some providers (e.g., LOCAL)
    local_config = ModelConfig(
        model_name="local-model",
        provider=ModelProvider.LOCAL
    )
    assert local_config.api_key is None
    
    # Should require API key for cloud providers (e.g., OPENAI)
    with pytest.raises(ValidationError):
        ModelConfig(
            model_name="gpt-4",
            provider=ModelProvider.OPENAI,
            api_key=None
        )


# ==================== TESTS FOR MEMORY CONFIG ====================

def test_memory_config_basic_instantiation(valid_memory_config):
    """Test that a memory config can be instantiated with valid data."""
    assert valid_memory_config.max_entries == 5000
    assert valid_memory_config.use_embeddings is True
    assert valid_memory_config.embedding_model == "text-embedding-ada-002"
    assert valid_memory_config.default_retention == MemoryRetentionPolicy.SESSION  # Default value


def test_memory_config_parameters():
    """Test memory config with parameters."""
    config = MemoryConfig(
        max_entries=10000,
        use_embeddings=True,
        embedding_model="text-embedding-3-large",
        embedding_dimensions=3072,
        similarity_threshold=0.85,
        default_retention=MemoryRetentionPolicy.PERMANENT,
        expiration_days=None,
        auto_summarize=True,
        summarization_threshold=100
    )
    
    assert config.max_entries == 10000
    assert config.use_embeddings is True
    assert config.embedding_model == "text-embedding-3-large"
    assert config.embedding_dimensions == 3072
    assert config.similarity_threshold == 0.85
    assert config.default_retention == MemoryRetentionPolicy.PERMANENT
    assert config.expiration_days is None
    assert config.auto_summarize is True
    assert config.summarization_threshold == 100


def test_memory_config_validation():
    """Test validation in memory config."""
    # Invalid max entries (negative)
    with pytest.raises(ValidationError):
        MemoryConfig(
            max_entries=-1,
            use_embeddings=True
        )
    
    # Invalid similarity threshold (too high)
    with pytest.raises(ValidationError):
        MemoryConfig(
            max_entries=1000,
            use_embeddings=True,
            similarity_threshold=1.1
        )
    
    # Invalid similarity threshold (too low)
    with pytest.raises(ValidationError):
        MemoryConfig(
            max_entries=1000,
            use_embeddings=True,
            similarity_threshold=-0.1
        )
    
    # Missing embedding model when use_embeddings is True
    with pytest.raises(ValidationError):
        MemoryConfig(
            max_entries=1000,
            use_embeddings=True,
            embedding_model=None
        )


def test_memory_config_to_dict(valid_memory_config):
    """Test conversion to dictionary."""
    config_dict = valid_memory_config.to_dict()
    
    assert isinstance(config_dict, dict)
    assert config_dict["max_entries"] == 5000
    assert config_dict["use_embeddings"] is True
    assert config_dict["embedding_model"] == "text-embedding-ada-002"
    assert config_dict["default_retention"] == "session"


# ==================== TESTS FOR AGENT CONFIG ====================

def test_agent_config_basic_instantiation(valid_agent_config, valid_model_config):
    """Test that an agent config can be instantiated with valid data."""
    assert valid_agent_config.name == "TestAgent"
    assert valid_agent_config.description == "A test agent"
    assert valid_agent_config.model.model_name == valid_model_config.model_name
    assert valid_agent_config.model.provider == valid_model_config.provider
    assert valid_agent_config.tools == ["search", "calculator"]
    assert valid_agent_config.memory is None  # Default value


def test_agent_config_with_memory():
    """Test agent config with memory configuration."""
    model_config = ModelConfig(
        model_name="gpt-4",
        provider=ModelProvider.OPENAI,
        api_key="sk-test123"
    )
    
    memory_config = MemoryConfig(
        max_entries=5000,
        use_embeddings=True,
        embedding_model="text-embedding-ada-002"
    )
    
    agent_config = AgentConfig(
        name="TestAgentWithMemory",
        description="A test agent with memory",
        model=model_config,
        memory=memory_config,
        tools=["search", "calculator"]
    )
    
    assert agent_config.name == "TestAgentWithMemory"
    assert agent_config.model.model_name == "gpt-4"
    assert agent_config.memory is not None
    assert agent_config.memory.max_entries == 5000
    assert agent_config.memory.use_embeddings is True


def test_agent_config_validation():
    """Test validation in agent config."""
    model_config = ModelConfig(
        model_name="gpt-4",
        provider=ModelProvider.OPENAI,
        api_key="sk-test123"
    )
    
    # Name can't be empty
    with pytest.raises(ValidationError):
        AgentConfig(
            name="",
            description="Agent with empty name",
            model=model_config
        )
    
    # Name can't be just whitespace
    with pytest.raises(ValidationError):
        AgentConfig(
            name="   ",
            description="Agent with whitespace name",
            model=model_config
        )
    
    # Description can't be empty if provided
    with pytest.raises(ValidationError):
        AgentConfig(
            name="TestAgent",
            description="",
            model=model_config
        )


def test_agent_config_to_dict(valid_agent_config):
    """Test conversion to dictionary."""
    config_dict = valid_agent_config.to_dict()
    
    assert isinstance(config_dict, dict)
    assert config_dict["name"] == "TestAgent"
    assert config_dict["description"] == "A test agent"
    assert config_dict["tools"] == ["search", "calculator"]
    assert isinstance(config_dict["model"], dict)
    assert config_dict["model"]["model_name"] == "gpt-4"
    assert config_dict["model"]["provider"] == "openai"
    assert config_dict["memory"] is None


def test_agent_config_integration():
    """Test integration between different config components."""
    model_config = ModelConfig(
        model_name="gpt-4-turbo",
        provider=ModelProvider.OPENAI,
        api_key="sk-test456",
        temperature=0.5
    )
    
    memory_config = MemoryConfig(
        max_entries=10000,
        use_embeddings=True,
        embedding_model="text-embedding-3-small",
        embedding_dimensions=1536,
        similarity_threshold=0.8
    )
    
    agent_config = AgentConfig(
        name="IntegratedTestAgent",
        description="An agent with integrated configs",
        model=model_config,
        memory=memory_config,
        tools=["search", "calculator", "weather"],
        system_prompt="You are a helpful assistant."
    )
    
    assert agent_config.name == "IntegratedTestAgent"
    
    # Model config integration
    assert agent_config.model.model_name == "gpt-4-turbo"
    assert agent_config.model.provider == ModelProvider.OPENAI
    assert agent_config.model.temperature == 0.5
    
    # Memory config integration
    assert agent_config.memory.max_entries == 10000
    assert agent_config.memory.embedding_model == "text-embedding-3-small"
    assert agent_config.memory.similarity_threshold == 0.8
    
    # Agent-specific configs
    assert agent_config.tools == ["search", "calculator", "weather"]
    assert agent_config.system_prompt == "You are a helpful assistant."
    
    # Test dictionary conversion maintains the integration
    config_dict = agent_config.to_dict()
    assert config_dict["model"]["model_name"] == "gpt-4-turbo"
    assert config_dict["memory"]["embedding_model"] == "text-embedding-3-small"

