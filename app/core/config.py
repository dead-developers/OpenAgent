import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings."""
    model_config = {"extra": "allow"}
    
    # Base settings
    APP_NAME: str = "OpenAgent UI"
    API_PREFIX: str = "/api"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Authentication settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "openagent_secret_key_change_in_production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    
    # CORS settings
    CORS_ORIGINS: List[str] = ["*"]
    
    # Database settings
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        f"sqlite:///{Path(__file__).parent.parent.parent / 'openagent.db'}"
    )
    
    # Default configuration
    DEFAULT_CONFIG: Dict[str, Any] = {
        "sections": [
            {
                "id": "general",
                "title": "General Settings",
                "description": "General configuration for OpenAgent",
                "fields": [
                    {
                        "id": "agent_type",
                        "label": "Agent Type",
                        "type": "select",
                        "options": [
                            {"value": "manus", "label": "Manus"},
                            {"value": "mcp", "label": "MCP"}
                        ],
                        "value": "manus",
                        "required": True
                    },
                    {
                        "id": "use_planning",
                        "label": "Use Planning",
                        "type": "boolean",
                        "value": True,
                        "description": "Whether to use planning for execution"
                    }
                ]
            },
            {
                "id": "api_keys",
                "title": "API Keys",
                "description": "API keys for various services",
                "fields": [
                    {
                        "id": "openai_api_key",
                        "label": "OpenAI API Key",
                        "type": "password",
                        "value": os.getenv("OPENAI_API_KEY", ""),
                        "description": "API key for OpenAI services"
                    },
                    {
                        "id": "huggingface_api_key",
                        "label": "HuggingFace API Key",
                        "type": "password",
                        "value": os.getenv("HUGGINGFACE_API_KEY", ""),
                        "description": "API key for HuggingFace services"
                    }
                ]
            },
            {
                "id": "model_settings",
                "title": "Model Settings",
                "description": "Configuration for language models",
                "fields": [
                    {
                        "id": "main_model",
                        "label": "Main Model",
                        "type": "select",
                        "options": [
                            {"value": "gpt-4", "label": "GPT-4"},
                            {"value": "gpt-4-turbo", "label": "GPT-4 Turbo"},
                            {"value": "gpt-3.5-turbo", "label": "GPT-3.5 Turbo"},
                            {"value": "claude-3-opus", "label": "Claude 3 Opus"},
                            {"value": "claude-3-sonnet", "label": "Claude 3 Sonnet"}
                        ],
                        "value": "gpt-4",
                        "required": True
                    },
                    {
                        "id": "planning_model",
                        "label": "Planning Model",
                        "type": "select",
                        "options": [
                            {"value": "gpt-4", "label": "GPT-4"},
                            {"value": "gpt-4-turbo", "label": "GPT-4 Turbo"},
                            {"value": "gpt-3.5-turbo", "label": "GPT-3.5 Turbo"},
                            {"value": "claude-3-opus", "label": "Claude 3 Opus"},
                            {"value": "claude-3-sonnet", "label": "Claude 3 Sonnet"}
                        ],
                        "value": "gpt-3.5-turbo",
                        "required": True
                    },
                    {
                        "id": "temperature",
                        "label": "Temperature",
                        "type": "number",
                        "value": 0.7,
                        "description": "Temperature for model generation",
                        "validation": {
                            "min": 0,
                            "max": 2,
                            "message": "Temperature must be between 0 and 2"
                        }
                    }
                ]
            }
        ]
    }
    
    model_config = {
        "extra": "allow",
        "env_file": ".env",
        "case_sensitive": True
    }

# Create global settings object
settings = Settings()
