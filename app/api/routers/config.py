from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import json

from app.api.dependencies import get_current_active_user, get_current_user
from app.api.schemas.config import (
    ConfigurationUpdate,
    ConfigurationResponse,
    ConfigPresetCreate,
    ConfigPresetResponse,
    ConfigPresetListResponse
)
from app.db.session import get_db
from app.db.models import Configuration, ConfigurationPreset, User
from app.core.config import settings
from app.logger import logger

router = APIRouter()

@router.get("", response_model=ConfigurationResponse)
async def get_configuration(
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user)
):
    """Get current configuration."""
    # If user is not authenticated, return default configuration
    if not current_user:
        return {
            "id": "default",
            "name": "Default Configuration",
            "description": "Default configuration for anonymous users",
            "is_active": True,
            "sections": settings.DEFAULT_CONFIG["sections"],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
    
    # Get active configuration for user
    config = db.query(Configuration).filter(
        Configuration.user_id == current_user.id,
        Configuration.is_active == True
    ).first()
    
    if not config:
        # Create default configuration if none exists
        config = Configuration(
            id=f"config_{uuid.uuid4().hex}",
            user_id=current_user.id,
            name="Default Configuration",
            description="Default configuration created automatically",
            is_active=True,
            data=json.dumps(settings.DEFAULT_CONFIG),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        db.add(config)
        db.commit()
    
    # Parse configuration data
    config_data = json.loads(config.data)
    
    return {
        "id": config.id,
        "name": config.name,
        "description": config.description,
        "is_active": config.is_active,
        "sections": config_data.get("sections", []),
        "created_at": config.created_at,
        "updated_at": config.updated_at
    }

@router.post("", response_model=ConfigurationResponse)
async def update_configuration(
    config_update: ConfigurationUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Update configuration."""
    # Get active configuration for user
    config = db.query(Configuration).filter(
        Configuration.user_id == current_user.id,
        Configuration.is_active == True
    ).first()
    
    if not config:
        # Create new configuration if none exists
        config = Configuration(
            id=f"config_{uuid.uuid4().hex}",
            user_id=current_user.id,
            name="Default Configuration",
            description="Default configuration created automatically",
            is_active=True,
            created_at=datetime.utcnow()
        )
        db.add(config)
    
    # Update configuration
    config.data = json.dumps({"sections": [section.dict() for section in config_update.sections]})
    config.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(config)
    
    # Parse configuration data
    config_data = json.loads(config.data)
    
    return {
        "id": config.id,
        "name": config.name,
        "description": config.description,
        "is_active": config.is_active,
        "sections": config_data.get("sections", []),
        "created_at": config.created_at,
        "updated_at": config.updated_at
    }

@router.get("/presets", response_model=ConfigPresetListResponse)
async def list_configuration_presets(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """List configuration presets."""
    # Get presets for user
    presets = db.query(ConfigurationPreset).filter(
        ConfigurationPreset.user_id == current_user.id
    ).order_by(
        ConfigurationPreset.is_default.desc(),
        ConfigurationPreset.name
    ).all()
    
    return {
        "presets": presets
    }

@router.get("/presets/{preset_id}", response_model=ConfigurationResponse)
async def get_configuration_preset(
    preset_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get configuration preset."""
    # Get preset
    preset = db.query(ConfigurationPreset).filter(
        ConfigurationPreset.id == preset_id,
        ConfigurationPreset.user_id == current_user.id
    ).first()
    
    if not preset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Preset not found"
        )
    
    # Parse preset data
    preset_data = json.loads(preset.data)
    
    return {
        "id": preset.id,
        "name": preset.name,
        "description": preset.description,
        "is_active": False,
        "sections": preset_data.get("sections", []),
        "created_at": preset.created_at,
        "updated_at": preset.updated_at
    }

@router.post("/presets", response_model=ConfigPresetResponse)
async def create_configuration_preset(
    preset_create: ConfigPresetCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Create configuration preset."""
    # Create preset
    preset = ConfigurationPreset(
        id=f"preset_{uuid.uuid4().hex}",
        user_id=current_user.id,
        name=preset_create.name,
        description=preset_create.description,
        is_default=preset_create.is_default,
        data=json.dumps({"sections": [section.dict() for section in preset_create.sections]}),
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    
    # If this is the default preset, unset other defaults
    if preset.is_default:
        db.query(ConfigurationPreset).filter(
            ConfigurationPreset.user_id == current_user.id,
            ConfigurationPreset.is_default == True
        ).update({"is_default": False})
    
    db.add(preset)
    db.commit()
    db.refresh(preset)
    
    return preset

@router.delete("/presets/{preset_id}", response_model=ConfigPresetResponse)
async def delete_configuration_preset(
    preset_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete configuration preset."""
    # Get preset
    preset = db.query(ConfigurationPreset).filter(
        ConfigurationPreset.id == preset_id,
        ConfigurationPreset.user_id == current_user.id
    ).first()
    
    if not preset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Preset not found"
        )
    
    # Cannot delete default preset
    if preset.is_default:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete default preset"
        )
    
    # Delete preset
    db.delete(preset)
    db.commit()
    
    return preset
