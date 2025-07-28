from pydantic import BaseModel, EmailStr
from typing import Optional

class UserBase(BaseModel):
    full_name: str
    email:    EmailStr

class UserCreate(UserBase):
    password: str

class UserLogin(BaseModel):
    email:    EmailStr
    password: str

class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    password:  Optional[str] = None

class UserOut(UserBase):
    id: str

    class Config:
        orm_mode = True
