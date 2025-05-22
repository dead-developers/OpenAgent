from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

class ExecutionCreate(BaseModel):
    """Schema for creating a new execution."""
    prompt: str = Field(..., description="The prompt to execute")
    use_planning: bool = Field(True, description="Whether to use planning for execution")
    agent_type: str = Field("manus", description="The type of agent to use")

class ExecutionResponse(BaseModel):
    """Schema for execution response."""
    id: str
    prompt: str
    result: Optional[str] = None
    status: str
    start_time: datetime
    end_time: Optional[datetime] = None
    agent_type: Optional[str] = None

    class Config:
        orm_mode = True

class ExecutionListResponse(BaseModel):
    """Schema for listing executions."""
    executions: List[ExecutionResponse]
    total: int
    skip: int
    limit: int

class ExecutionStepResponse(BaseModel):
    """Schema for execution step response."""
    id: str
    execution_id: str
    step_number: int
    tool_name: Optional[str] = None
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    status: str
    execution_time: Optional[int] = None
    timestamp: datetime

    class Config:
        orm_mode = True

class PlanResponse(BaseModel):
    """Schema for plan response."""
    id: str
    execution_id: str
    title: str
    steps: List[str]
    step_statuses: List[str]
    step_notes: Optional[List[str]] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class WebSocketMessage(BaseModel):
    """Schema for WebSocket messages."""
    type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    execution_id: str
    data: Dict[str, Any]
