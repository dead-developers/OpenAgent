from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.db.session import Base

class User(Base):
    """User model for authentication and authorization."""
    __tablename__ = "users"

    id = Column(String(50), primary_key=True, index=True)
    email = Column(String(100), unique=True, index=True)
    hashed_password = Column(String(100))
    full_name = Column(String(100))
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    executions = relationship("Execution", back_populates="user")
    configurations = relationship("Configuration", back_populates="user")
    config_presets = relationship("ConfigurationPreset", back_populates="user")

class Execution(Base):
    """Execution model for tracking agent executions."""
    __tablename__ = "executions"

    id = Column(String(50), primary_key=True, index=True)
    user_id = Column(String(50), ForeignKey("users.id"), nullable=True)
    prompt = Column(Text)
    result = Column(Text, nullable=True)
    status = Column(String(20))  # running, completed, error
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    agent_type = Column(String(50), nullable=True)  # manus, mcp, etc.
    
    # Relationships
    user = relationship("User", back_populates="executions")
    steps = relationship("ExecutionStep", back_populates="execution", cascade="all, delete-orphan")
    plans = relationship("Plan", back_populates="execution", cascade="all, delete-orphan")

class ExecutionStep(Base):
    """Execution step model for tracking individual steps in an execution."""
    __tablename__ = "execution_steps"

    id = Column(String(50), primary_key=True, index=True, default=lambda: f"step_{uuid.uuid4().hex}")
    execution_id = Column(String(50), ForeignKey("executions.id"))
    step_number = Column(Integer)
    tool_name = Column(String(100), nullable=True)
    input_data = Column(JSON, nullable=True)
    output_data = Column(JSON, nullable=True)
    status = Column(String(20), default="success")  # success, error
    execution_time = Column(Integer, nullable=True)  # in milliseconds
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    execution = relationship("Execution", back_populates="steps")

class Plan(Base):
    """Plan model for tracking execution plans."""
    __tablename__ = "plans"

    id = Column(String(50), primary_key=True, index=True)
    execution_id = Column(String(50), ForeignKey("executions.id"))
    title = Column(String(200))
    steps = Column(JSON)  # List of step descriptions
    step_statuses = Column(JSON)  # List of step statuses
    step_notes = Column(JSON, nullable=True)  # List of step notes
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    execution = relationship("Execution", back_populates="plans")

class Configuration(Base):
    """Configuration model for storing user configurations."""
    __tablename__ = "configurations"

    id = Column(String(50), primary_key=True, index=True)
    user_id = Column(String(50), ForeignKey("users.id"))
    name = Column(String(100))
    description = Column(Text, nullable=True)
    data = Column(JSON)  # JSON configuration data
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="configurations")

class ConfigurationPreset(Base):
    """Configuration preset model for storing reusable configurations."""
    __tablename__ = "configuration_presets"

    id = Column(String(50), primary_key=True, index=True)
    user_id = Column(String(50), ForeignKey("users.id"))
    name = Column(String(100))
    description = Column(Text, nullable=True)
    data = Column(JSON)  # JSON configuration data
    is_default = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="config_presets")
