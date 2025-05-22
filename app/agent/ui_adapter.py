from typing import Dict, Any, Optional, List, Callable
import asyncio
import json
import uuid
from datetime import datetime

from app.db.session import SessionLocal
from app.db.models import Execution, ExecutionStep, Plan
from app.agent.manus import Manus
from app.agent.mcp import MCPAgent
from app.flow.flow_factory import FlowFactory, FlowType
from app.logger import logger
from app.api.routers.websockets import broadcast_message

# Store for active websocket connections
websocket_clients = {}

# Register a websocket client
def register_client(client_id: str, websocket):
    websocket_clients[client_id] = websocket
    logger.info(f"Registered WebSocket client: {client_id}")

# Unregister a websocket client
def unregister_client(client_id: str):
    if client_id in websocket_clients:
        del websocket_clients[client_id]
        logger.info(f"Unregistered WebSocket client: {client_id}")

# Create a UI-adapted Manus agent
class UIAdaptedManus(Manus):
    """A version of Manus agent that reports execution progress to the UI"""
    
    execution_id: str = None
    
    @classmethod
    async def create(cls, execution_id: Optional[str] = None):
        """Create and initialize a new Manus agent with UI reporting"""
        agent = cls()
        if execution_id:
            agent.execution_id = execution_id
        await agent.initialize()
        return agent
    
    async def run(self, request: Optional[str] = None) -> str:
        """Override run to capture execution progress"""
        if not self.execution_id:
            self.execution_id = f"exec_{uuid.uuid4().hex}"
            
        # Create execution record in database
        db = SessionLocal()
        try:
            execution = Execution(
                id=self.execution_id,
                prompt=request,
                start_time=datetime.utcnow(),
                status="running",
                agent_type="manus"
            )
            db.add(execution)
            db.commit()
        finally:
            db.close()
            
        # Broadcast execution start
        await broadcast_message("execution_started", {
            "execution": {
                "id": self.execution_id,
                "prompt": request,
                "status": "running",
                "start_time": datetime.utcnow().isoformat()
            }
        }, self.execution_id)
        
        try:
            # Run the agent
            result = await super().run(request)
            
            # Update execution record on completion
            db = SessionLocal()
            try:
                execution = db.query(Execution).filter(Execution.id == self.execution_id).first()
                if execution:
                    execution.status = "completed"
                    execution.end_time = datetime.utcnow()
                    execution.result = result
                    db.commit()
            finally:
                db.close()
                
            # Broadcast execution completion
            await broadcast_message("execution_completed", {
                "execution": {
                    "id": self.execution_id,
                    "status": "completed",
                    "end_time": datetime.utcnow().isoformat(),
                    "result": result
                }
            }, self.execution_id)
            
            return result
        except Exception as e:
            # Update execution record on error
            db = SessionLocal()
            try:
                execution = db.query(Execution).filter(Execution.id == self.execution_id).first()
                if execution:
                    execution.status = "error"
                    execution.end_time = datetime.utcnow()
                    execution.result = f"Error: {str(e)}"
                    db.commit()
            finally:
                db.close()
                
            # Broadcast execution error
            await broadcast_message("execution_error", {
                "execution": {
                    "id": self.execution_id,
                    "status": "error",
                    "end_time": datetime.utcnow().isoformat(),
                    "error": str(e)
                }
            }, self.execution_id)
            
            raise
    
    async def think(self) -> bool:
        """Override think to capture thinking steps"""
        result = await super().think()
        
        # Capture the last message (thinking)
        if self.memory.messages and self.execution_id:
            last_message = self.memory.messages[-1]
            if last_message.role == "assistant":
                # Broadcast thinking update
                await broadcast_message("thinking_update", {
                    "content": last_message.content
                }, self.execution_id)
                
        return result
    
    async def _handle_tool_call(self, tool_call: Dict[str, Any]) -> Any:
        """Override tool call handling to capture tool execution"""
        start_time = datetime.utcnow()
        
        # Execute the tool call
        result = await super()._handle_tool_call(tool_call)
        
        # Calculate execution time
        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000  # in milliseconds
        
        if self.execution_id:
            # Extract tool name and parameters
            tool_name = tool_call.get("name", "unknown")
            parameters = tool_call.get("parameters", {})
            
            # Create tool execution record
            db = SessionLocal()
            try:
                step = ExecutionStep(
                    execution_id=self.execution_id,
                    step_number=self.current_step,
                    tool_name=tool_name,
                    input_data=parameters,
                    output_data=result if isinstance(result, dict) else {"result": str(result)},
                    status="success",
                    execution_time=execution_time,
                    timestamp=datetime.utcnow()
                )
                db.add(step)
                db.commit()
                
                # Get the step ID
                step_id = step.id
            finally:
                db.close()
                
            # Broadcast tool execution
            await broadcast_message("step_update", {
                "step": {
                    "id": step_id,
                    "execution_id": self.execution_id,
                    "step_number": self.current_step,
                    "tool_name": tool_name,
                    "input_data": parameters,
                    "output_data": result if isinstance(result, dict) else {"result": str(result)},
                    "status": "success",
                    "execution_time": execution_time,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }, self.execution_id)
        
        return result

# Create a UI-adapted MCP agent
class UIAdaptedMCPAgent(MCPAgent):
    """A version of MCP agent that reports execution progress to the UI"""
    
    execution_id: str = None
    
    async def run(self, request: Optional[str] = None) -> str:
        """Override run to capture execution progress"""
        if not self.execution_id:
            self.execution_id = f"exec_{uuid.uuid4().hex}"
            
        # Create execution record in database
        db = SessionLocal()
        try:
            execution = Execution(
                id=self.execution_id,
                prompt=request,
                start_time=datetime.utcnow(),
                status="running",
                agent_type="mcp"
            )
            db.add(execution)
            db.commit()
        finally:
            db.close()
            
        # Broadcast execution start
        await broadcast_message("execution_started", {
            "execution": {
                "id": self.execution_id,
                "prompt": request,
                "status": "running",
                "start_time": datetime.utcnow().isoformat()
            }
        }, self.execution_id)
        
        try:
            # Run the agent
            result = await super().run(request)
            
            # Update execution record on completion
            db = SessionLocal()
            try:
                execution = db.query(Execution).filter(Execution.id == self.execution_id).first()
                if execution:
                    execution.status = "completed"
                    execution.end_time = datetime.utcnow()
                    execution.result = result
                    db.commit()
            finally:
                db.close()
                
            # Broadcast execution completion
            await broadcast_message("execution_completed", {
                "execution": {
                    "id": self.execution_id,
                    "status": "completed",
                    "end_time": datetime.utcnow().isoformat(),
                    "result": result
                }
            }, self.execution_id)
            
            return result
        except Exception as e:
            # Update execution record on error
            db = SessionLocal()
            try:
                execution = db.query(Execution).filter(Execution.id == self.execution_id).first()
                if execution:
                    execution.status = "error"
                    execution.end_time = datetime.utcnow()
                    execution.result = f"Error: {str(e)}"
                    db.commit()
            finally:
                db.close()
                
            # Broadcast execution error
            await broadcast_message("execution_error", {
                "execution": {
                    "id": self.execution_id,
                    "status": "error",
                    "end_time": datetime.utcnow().isoformat(),
                    "error": str(e)
                }
            }, self.execution_id)
            
            raise
    
    async def think(self) -> bool:
        """Override think to capture thinking steps"""
        result = await super().think()
        
        # Capture the last message (thinking)
        if self.memory.messages and self.execution_id:
            last_message = self.memory.messages[-1]
            if last_message.role == "assistant":
                # Broadcast thinking update
                await broadcast_message("thinking_update", {
                    "content": last_message.content
                }, self.execution_id)
                
        return result

# Create a UI-adapted Planning Flow
class UIAdaptedPlanningFlow:
    """A version of PlanningFlow that reports planning progress to the UI"""
    
    def __init__(self, agents: Dict[str, Any], execution_id: Optional[str] = None):
        self.flow = FlowFactory.create_flow(FlowType.PLANNING, agents)
        self.execution_id = execution_id
        
        # Monkey patch the flow's _create_initial_plan method
        original_create_plan = self.flow._create_initial_plan
        
        async def patched_create_plan(request: str) -> None:
            await original_create_plan(request)
            await self._on_plan_created()
            
        self.flow._create_initial_plan = patched_create_plan
        
        # Monkey patch the flow's _mark_step_completed method
        original_mark_completed = self.flow._mark_step_completed
        
        async def patched_mark_completed() -> None:
            await original_mark_completed()
            await self._on_plan_updated()
            
        self.flow._mark_step_completed = patched_mark_completed
    
    async def execute(self, request: str) -> str:
        """Execute the flow with the given request"""
        return await self.flow.execute(request)
    
    async def _on_plan_created(self) -> None:
        """Handle plan creation event"""
        if not self.execution_id:
            return
            
        # Get the plan data
        plan_data = self.flow.planning_tool.plans.get(self.flow.active_plan_id)
        if not plan_data:
            return
            
        # Create plan record
        db = SessionLocal()
        try:
            plan = Plan(
                id=self.flow.active_plan_id,
                execution_id=self.execution_id,
                title=plan_data.get("title", "Untitled Plan"),
                steps=plan_data.get("steps", []),
                step_statuses=plan_data.get("step_statuses", []),
                step_notes=plan_data.get("step_notes", []),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            db.add(plan)
            db.commit()
        finally:
            db.close()
            
        # Broadcast plan creation
        await broadcast_message("plan_update", {
            "plan": {
                "id": self.flow.active_plan_id,
                "execution_id": self.execution_id,
                "title": plan_data.get("title", "Untitled Plan"),
                "steps": plan_data.get("steps", []),
                "step_statuses": plan_data.get("step_statuses", []),
                "step_notes": plan_data.get("step_notes", []),
                "created_at": datetime.utcnow().isoformat()
            }
        }, self.execution_id)
    
    async def _on_plan_updated(self) -> None:
        """Handle plan update event"""
        if not self.execution_id or not self.flow.active_plan_id:
            return
            
        # Get the updated plan data
        plan_data = self.flow.planning_tool.plans.get(self.flow.active_plan_id)
        if not plan_data:
            return
            
        # Update plan record
        db = SessionLocal()
        try:
            plan = db.query(Plan).filter(Plan.id == self.flow.active_plan_id).first()
            if plan:
                plan.steps = plan_data.get("steps", [])
                plan.step_statuses = plan_data.get("step_statuses", [])
                plan.step_notes = plan_data.get("step_notes", [])
                plan.updated_at = datetime.utcnow()
                db.commit()
        finally:
            db.close()
            
        # Calculate progress
        step_statuses = plan_data.get("step_statuses", [])
        total_steps = len(step_statuses)
        completed_steps = step_statuses.count("completed") if total_steps > 0 else 0
        progress = (completed_steps / total_steps) * 100 if total_steps > 0 else 0
            
        # Broadcast plan update
        await broadcast_message("plan_update", {
            "plan": {
                "id": self.flow.active_plan_id,
                "execution_id": self.execution_id,
                "steps": plan_data.get("steps", []),
                "step_statuses": plan_data.get("step_statuses", []),
                "step_notes": plan_data.get("step_notes", []),
                "progress": progress,
                "updated_at": datetime.utcnow().isoformat()
            }
        }, self.execution_id)

# Factory function to create UI-adapted agents
def create_ui_adapted_agent(agent_type: str = "manus", execution_id: str = None) -> Any:
    """Create an agent with UI reporting capabilities"""
    if agent_type.lower() == "manus":
        agent = UIAdaptedManus()
        if execution_id:
            agent.execution_id = execution_id
        return agent
    elif agent_type.lower() == "mcp":
        agent = UIAdaptedMCPAgent()
        if execution_id:
            agent.execution_id = execution_id
        return agent
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")

# Factory function to create UI-adapted flows
def create_ui_adapted_flow(flow_type: str = "planning", agents: Dict[str, Any] = None, execution_id: str = None) -> Any:
    """Create a flow with UI reporting capabilities"""
    if flow_type.lower() == "planning":
        flow = UIAdaptedPlanningFlow(agents or {}, execution_id)
        return flow
    else:
        raise ValueError(f"Unsupported flow type: {flow_type}")
