from pydantic import BaseModel, Field, EmailStr
from typing import Optional
from datetime import datetime

class Token(BaseModel):
    """Schema for authentication token."""
    access_token: str
    token_type: str

class TokenPayload(BaseModel):
    """Schema for token payload."""
    sub: Optional[str] = None
    exp: Optional[datetime] = None

class UserBase(BaseModel):
    """Base schema for user."""
    email: EmailStr
    full_name: Optional[str] = None
    is_active: Optional[bool] = True
    is_admin: Optional[bool] = False

class UserCreate(UserBase):
    """Schema for creating a new user."""
    password: str = Field(..., min_length=8)

class UserUpdate(UserBase):
    """Schema for updating a user."""
    password: Optional[str] = Field(None, min_length=8)

class UserResponse(UserBase):
    """Schema for user response."""
    id: str
    created_at: datetime

    class Config:
        orm_mode = True
