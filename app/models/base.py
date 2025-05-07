"""
Base models for the OpenAgent system.

This module contains the core Pydantic models that form the foundation
of the OpenAgent system, including models for memory entries, agent actions,
agent thoughts, and configuration.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


class MemoryType(str, Enum):
    """Types of memory entries"""
    
    CONVERSATION = "conversation"  # Message exchanges
    ACTION = "action"              # Agent actions
    THOUGHT = "thought"            # Agent thoughts/reasoning
    OBSERVATION = "observation"    # Results from actions
    PLAN = "plan"                  # Plans created by the agent
    METADATA = "metadata"          # System metadata
    EXTERNAL = "external"          # External knowledge/context


class ExecutionStatus(str, Enum):
    """Status of an action or task execution"""
    
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class MemoryRetentionPolicy(str, Enum):
    """Memory retention policies"""
    
    PERMANENT = "permanent"  # Never expire
    SESSION = "session"      # Retain for the current session only
    TEMPORARY = "temporary"  # Short-term, can be pruned when memory is full
    EXPIRING = "expiring"    # Expires after a set time


class MemoryEntry(BaseModel):
    """
    A structured memory entry in the OpenAgent system.
    
    Memory entries represent discrete pieces of information that the
    agent stores and can retrieve later, including conversations,
    observations, actions taken, and internal thoughts.
    """
    
    id: UUID = Field(default_factory=uuid4)
    type: MemoryType
    content: Any  # Can be any type, depends on the memory type
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    retention: MemoryRetentionPolicy = MemoryRetentionPolicy.SESSION
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    session_id: Optional[UUID] = None
    
    # Optional fields based on memory type
    related_entries: List[UUID] = Field(default_factory=list)
    expiration: Optional[datetime] = None
    embedding: Optional[List[float]] = None
    
    @field_validator('embedding')
    @classmethod
    def validate_embedding(cls, v):
        """Validate that embeddings are normalized if present"""
        if v is not None:
            if not all(isinstance(x, float) for x in v):
                raise ValueError("All embedding values must be floats")
        return v
    
    @model_validator(mode='after')
    def validate_memory_entry(self):
        """Validate the complete memory entry based on its type"""
        # For CONVERSATION type, content should be a message or list of messages
        if self.type == MemoryType.CONVERSATION:
            if not (isinstance(self.content, dict) or isinstance(self.content, list)):
                raise ValueError("Conversation memory must have content as message dict or list")
        
        # If retention is EXPIRING, expiration must be set
        if self.retention == MemoryRetentionPolicy.EXPIRING and self.expiration is None:
            raise ValueError("Expiring memories must have an expiration date")
            
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary form"""
        return {
            "id": str(self.id),
            "type": self.type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "tags": self.tags,
            "retention": self.retention.value,
            "importance": self.importance,
            "session_id": str(self.session_id) if self.session_id else None,
            "related_entries": [str(entry_id) for entry_id in self.related_entries],
            "expiration": self.expiration.isoformat() if self.expiration else None,
        }


class ActionType(str, Enum):
    """Types of actions an agent can take"""
    
    TOOL_CALL = "tool_call"       # External tool usage
    API_CALL = "api_call"         # Call to external API
    INTERNAL = "internal"         # Internal state change
    MESSAGE = "message"           # Send a message
    QUERY = "query"               # Query for information
    DECISION = "decision"         # Make a decision
    PLANNING = "planning"         # Create or modify plans


class ActionParameter(BaseModel):
    """Parameter for an agent action"""
    
    name: str
    value: Any
    description: Optional[str] = None
    type_hint: Optional[str] = None


class AgentAction(BaseModel):
    """
    Represents an action taken by an agent.
    
    Actions are the primary way that agents interact with their environment,
    whether through tool calls, API requests, or message generation.
    """
    
    id: UUID = Field(default_factory=uuid4)
    type: ActionType
    name: str
    description: Optional[str] = None
    parameters: List[ActionParameter] = Field(default_factory=list)
    result: Optional[Any] = None
    status: ExecutionStatus = ExecutionStatus.PENDING
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    execution_time: Optional[float] = None  # Duration in milliseconds
    parent_action_id: Optional[UUID] = None
    session_id: Optional[UUID] = None
    memory_id: Optional[UUID] = None  # Reference to a memory entry
    
    @model_validator(mode='after')
    def validate_action(self):
        """Validate the action based on its type and status"""
        # If status is COMPLETED, result should be present
        if self.status == ExecutionStatus.COMPLETED and self.result is None:
            raise ValueError("Completed actions must have a result")
            
        # If status is FAILED, error should be present
        if self.status == ExecutionStatus.FAILED and self.error is None:
            raise ValueError("Failed actions must have an error message")
            
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary form"""
        return {
            "id": str(self.id),
            "type": self.type.value,
            "name": self.name,
            "description": self.description,
            "parameters": [param.model_dump() for param in self.parameters],
            "result": self.result,
            "status": self.status.value,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
            "execution_time": self.execution_time,
            "parent_action_id": str(self.parent_action_id) if self.parent_action_id else None,
            "session_id": str(self.session_id) if self.session_id else None,
            "memory_id": str(self.memory_id) if self.memory_id else None,
        }
    
    def with_result(self, result: Any, execution_time: Optional[float] = None) -> "AgentAction":
        """Create a new action with results set"""
        return self.model_copy(
            update={
                "result": result,
                "status": ExecutionStatus.COMPLETED,
                "execution_time": execution_time,
            }
        )
    
    def with_error(self, error: str, execution_time: Optional[float] = None) -> "AgentAction":
        """Create a new action with error set"""
        return self.model_copy(
            update={
                "error": error,
                "status": ExecutionStatus.FAILED,
                "execution_time": execution_time,
            }
        )


class ThoughtType(str, Enum):
    """Types of agent thoughts"""
    
    REASONING = "reasoning"       # Logical deduction or inference
    PLANNING = "planning"         # Planning next steps
    EVALUATION = "evaluation"     # Evaluating options or results
    REFLECTION = "reflection"     # Self-reflection on past actions
    CREATIVITY = "creativity"     # Creative idea generation
    SUMMARY = "summary"           # Summarization of information
    QUESTION = "question"         # Question formulation


class AgentThought(BaseModel):
    """
    Represents a thought or reasoning process of an agent.
    
    Thoughts capture the agent's internal decision-making process
    and reasoning, providing transparency into how and why the agent
    takes certain actions.
    """
    
    id: UUID = Field(default_factory=uuid4)
    type: ThoughtType
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    supporting_facts: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    related_action_ids: List[UUID] = Field(default_factory=list)
    related_thought_ids: List[UUID] = Field(default_factory=list)
    session_id: Optional[UUID] = None
    memory_id: Optional[UUID] = None  # Reference to a memory entry
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v):
        """Validate thought content"""
        if not v.strip():
            raise ValueError("Thought content cannot be empty")
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary form"""
        return {
            "id": str(self.id),
            "type": self.type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "supporting_facts": self.supporting_facts,
            "confidence": self.confidence,
            "related_action_ids": [str(action_id) for action_id in self.related_action_ids],
            "related_thought_ids": [str(thought_id) for thought_id in self.related_thought_ids],
            "session_id": str(self.session_id) if self.session_id else None,
            "memory_id": str(self.memory_id) if self.memory_id else None,
        }


class ModelProvider(str, Enum):
    """Supported LLM providers"""
    
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    GOOGLE = "google"
    OPEN_SOURCE = "open_source"
    OPENROUTER = "openrouter"
    FIREWORKS = "fireworks"
    LOCAL = "local"
    CUSTOM = "custom"


class ModelConfig(BaseModel):
    """Configuration for a language model"""
    
    model_name: str
    provider: ModelProvider
    api_key: Optional[str] = None
    api_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 60  # seconds
    retry_count: int = 3
    context_window: Optional[int] = None
    
    @model_validator(mode='after')
    def validate_model_config(self):
        """Validate model configuration"""
        # If provider is not LOCAL, api_key should be provided
        if (
            self.provider != ModelProvider.LOCAL 
            and self.provider != ModelProvider.OPEN_SOURCE
            and self.api_key is None
        ):
            raise ValueError(f"API key is required for {self.provider} provider")
        return self


class MemoryConfig(BaseModel):
    """Configuration for agent memory system"""
    
    max_entries: int = 1000
    use_embeddings: bool = True
    embedding_model: Optional[str] = None
    vector_db_connection: Optional[str] = None
    default_retention: MemoryRetentionPolicy = MemoryRetentionPolicy.SESSION
    importance_threshold: float = 0.3  # Minimum importance to keep in long-term memory
    

class AgentConfig(BaseModel):
    """Base configuration for an agent"""
    
    name: str
    description: Optional[str] = None
    model: ModelConfig
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    max_recursion_depth: int = 5
    max_execution_time: int = 300  # seconds
    verbose: bool = False
    tools: List[str] = Field(default_factory=list)
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        """Validate agent name"""
        if not v.strip():
            raise ValueError("Agent name cannot be empty")
        return v

