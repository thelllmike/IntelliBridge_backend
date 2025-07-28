from fastapi import APIRouter, HTTPException, status, Depends
import motor
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
from jose import jwt
from datetime import datetime, timedelta
from bson import ObjectId
import os

from schemas import UserCreate, UserLogin, UserUpdate, UserOut

router = APIRouter(prefix="/users", tags=["users"])

# ── CONFIG ───────────────────────────────────────────────────────────────────
MONGODB_URI = "mongodb+srv://yuvinsanketh10:EPveklyFAO7CeP3N@cluster0.5jwuszj.mongodb.net/job_analysis_db?retryWrites=true&w=majority&appName=Cluster0"

client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URI)

client     = AsyncIOMotorClient(MONGODB_URI)
db         = client["job_analysis_db"]    # your DB name
users_col  = db["users"]

pwd_ctx    = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = os.getenv("SECRET_KEY", "CHANGE_ME")
ALGORITHM  = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# ── HELPERS ──────────────────────────────────────────────────────────────────
def hash_password(pw: str) -> str:
    return pwd_ctx.hash(pw)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)

def create_access_token(subject: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = {"sub": subject, "exp": expire}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_user_by_email(email: str):
    return await users_col.find_one({"email": email})

async def get_user_by_id(user_id: str):
    doc = await users_col.find_one({"_id": ObjectId(user_id)})
    return doc

# ── ROUTES ───────────────────────────────────────────────────────────────────
@router.post("/register", response_model=UserOut, status_code=201)
async def register(user_in: UserCreate):
    if await get_user_by_email(user_in.email):
        raise HTTPException(400, "Email already registered")
    hashed = hash_password(user_in.password)
    doc = {
        "full_name":  user_in.full_name,
        "email":      user_in.email,
        "password":   hashed,
        "created_at": datetime.utcnow()
    }
    result = await users_col.insert_one(doc)
    return UserOut(id=str(result.inserted_id),
                   full_name=doc["full_name"],
                   email=doc["email"])

# @router.post("/login")
# async def login(data: UserLogin):
#     user = await get_user_by_email(data.email)
#     if not user or not verify_password(data.password, user["password"]):
#         raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
#                             detail="Invalid credentials")
#     token = create_access_token(str(user["_id"]))
#     return {"access_token": token, "token_type": "bearer"}

@router.post("/login")
async def login(data: UserLogin):
    user = await get_user_by_email(data.email)
    if not user or not verify_password(data.password, user["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    return {"user_id": str(user["_id"])}

@router.get("/{user_id}", response_model=UserOut)
async def read_user(user_id: str):
    user = await get_user_by_id(user_id)
    if not user:
        raise HTTPException(404, "User not found")
    return UserOut(id=str(user["_id"]),
                   full_name=user["full_name"],
                   email=user["email"])

@router.put("/{user_id}", response_model=UserOut)
async def update_user(user_id: str, upd: UserUpdate):
    user = await get_user_by_id(user_id)
    if not user:
        raise HTTPException(404, "User not found")

    update_data = {}
    if upd.full_name:
        update_data["full_name"] = upd.full_name
    if upd.password:
        update_data["password"] = hash_password(upd.password)

    if update_data:
        await users_col.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": update_data}
        )

    user = await get_user_by_id(user_id)
    return UserOut(id=str(user["_id"]),
                   full_name=user["full_name"],
                   email=user["email"])

@router.delete("/{user_id}", status_code=204)
async def delete_user(user_id: str):
    result = await users_col.delete_one({"_id": ObjectId(user_id)})
    if result.deleted_count == 0:
        raise HTTPException(404, "User not found")
