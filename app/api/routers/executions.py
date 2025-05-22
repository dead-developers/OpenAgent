from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid
from datetime import datetime
import asyncio

from app.api.dependencies import get_current_active_user, get_current_user
from app.api.schemas.execution import (
    ExecutionCreate,
    ExecutionResponse,
    ExecutionListResponse,
    ExecutionStepResponse,
    PlanResponse
)
from app.db.session import get_db
from app.db.models import Execution, ExecutionStep, Plan, User
from app.agent.ui_adapter import create_ui_adapted_agent, create_ui_adapted_flow
from app.logger import logger

router = APIRouter()

@router.post("", response_model=ExecutionResponse)
async def create_execution(
    execution: ExecutionCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user)
):
    """Create a new execution."""
    # Generate execution ID
    execution_id = f"exec_{uuid.uuid4().hex}"
    
    # Create execution record
    db_execution = Execution(
        id=execution_id,
        user_id=current_user.id if current_user else None,
        prompt=execution.prompt,
        status="running",
        start_time=datetime.utcnow(),
        agent_type=execution.agent_type
    )
    db.add(db_execution)
    db.commit()
    
    # Run execution in background
    background_tasks.add_task(
        run_execution_background,
        execution_id=execution_id,
        prompt=execution.prompt,
        use_planning=execution.use_planning,
        agent_type=execution.agent_type
    )
    
    return db_execution

async def run_execution_background(
    execution_id: str,
    prompt: str,
    use_planning: bool,
    agent_type: str
):
    """Run execution in background."""
    try:
        # Get database session
        db = next(get_db())
        
        if use_planning:
            # Create agent
            agent = create_ui_adapted_agent(agent_type, execution_id)
            
            # Create flow with planning
            agents = {agent_type: agent}
            flow = create_ui_adapted_flow("planning", agents, execution_id)
            
            # Execute flow
            result = await flow.execute(prompt)
        else:
            # Create and initialize agent
            agent = create_ui_adapted_agent(agent_type, execution_id)
            
            # Run agent
            result = await agent.run(prompt)
        
        # Update execution record
        execution = db.query(Execution).filter(Execution.id == execution_id).first()
        if execution:
            execution.status = "completed"
            execution.result = result
            execution.end_time = datetime.utcnow()
            db.commit()
    except Exception as e:
        logger.error(f"Error running execution {execution_id}: {str(e)}")
        
        # Update execution record with error
        try:
            db = next(get_db())
            execution = db.query(Execution).filter(Execution.id == execution_id).first()
            if execution:
                execution.status = "error"
                execution.result = f"Error: {str(e)}"
                execution.end_time = datetime.utcnow()
                db.commit()
        except Exception as db_error:
            logger.error(f"Error updating execution record: {str(db_error)}")

@router.get("", response_model=ExecutionListResponse)
async def list_executions(
    skip: int = 0,
    limit: int = 10,
    status: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user)
):
    """List executions."""
    # Build query
    query = db.query(Execution)
    
    # Filter by user if authenticated
    if current_user:
        query = query.filter(Execution.user_id == current_user.id)
    
    # Filter by status if provided
    if status:
        query = query.filter(Execution.status == status)
    
    # Get total count
    total = query.count()
    
    # Apply pagination
    query = query.order_by(Execution.start_time.desc()).offset(skip).limit(limit)
    
    # Get executions
    executions = query.all()
    
    return {
        "executions": executions,
        "total": total,
        "skip": skip,
        "limit": limit
    }

@router.get("/{execution_id}", response_model=ExecutionResponse)
async def get_execution(
    execution_id: str,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user)
):
    """Get execution details."""
    # Build query
    query = db.query(Execution).filter(Execution.id == execution_id)
    
    # Filter by user if authenticated
    if current_user:
        query = query.filter(Execution.user_id == current_user.id)
    
    # Get execution
    execution = query.first()
    
    if not execution:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Execution not found"
        )
    
    return execution

@router.get("/{execution_id}/steps", response_model=List[ExecutionStepResponse])
async def get_execution_steps(
    execution_id: str,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user)
):
    """Get execution steps."""
    # Check if execution exists and belongs to user
    query = db.query(Execution).filter(Execution.id == execution_id)
    
    # Filter by user if authenticated
    if current_user:
        query = query.filter(Execution.user_id == current_user.id)
    
    execution = query.first()
    
    if not execution:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Execution not found"
        )
    
    # Get steps
    steps = db.query(ExecutionStep).filter(
        ExecutionStep.execution_id == execution_id
    ).order_by(ExecutionStep.step_number).all()
    
    return steps

@router.get("/{execution_id}/plans", response_model=List[PlanResponse])
async def get_execution_plans(
    execution_id: str,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user)
):
    """Get execution plans."""
    # Check if execution exists and belongs to user
    query = db.query(Execution).filter(Execution.id == execution_id)
    
    # Filter by user if authenticated
    if current_user:
        query = query.filter(Execution.user_id == current_user.id)
    
    execution = query.first()
    
    if not execution:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Execution not found"
        )
    
    # Get plans
    plans = db.query(Plan).filter(
        Plan.execution_id == execution_id
    ).order_by(Plan.created_at).all()
    
    return plans
