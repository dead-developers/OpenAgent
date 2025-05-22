from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class ConfigField(BaseModel):
    """Schema for configuration field."""
    id: str
    label: str
    type: str  # text, number, boolean, select, password
    description: Optional[str] = None
    options: Optional[List[Dict[str, str]]] = None
    value: Any
    required: Optional[bool] = False
    validation: Optional[Dict[str, Any]] = None

class ConfigSection(BaseModel):
    """Schema for configuration section."""
    id: str
    title: str
    description: str
    fields: List[ConfigField]

class ConfigurationUpdate(BaseModel):
    """Schema for updating configuration."""
    sections: List[ConfigSection]

class ConfigurationResponse(BaseModel):
    """Schema for configuration response."""
    id: str
    name: str
    description: Optional[str] = None
    is_active: bool
    sections: List[ConfigSection]
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class ConfigPresetCreate(BaseModel):
    """Schema for creating configuration preset."""
    name: str
    description: Optional[str] = None
    is_default: bool = False
    sections: List[ConfigSection]

class ConfigPresetResponse(BaseModel):
    """Schema for configuration preset response."""
    id: str
    name: str
    description: Optional[str] = None
    is_default: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

class ConfigPresetListResponse(BaseModel):
    """Schema for listing configuration presets."""
    presets: List[ConfigPresetResponse]
