from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from mangum import Mangum
from jose import jwt, JWTError  # JWT creation and validation
import os

load_dotenv()

app = FastAPI()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

JWT_SECRET = os.getenv("JWT_SECRET")
ALGORITHM = "HS256"  # signing algorithm

# valid API keys (in production these would live in DynamoDB)
VALID_API_KEYS = {"test-key-123", "my-api-key"}

security = HTTPBearer()  # tells FastAPI to expect a Bearer token in the header

class ChatRequest(BaseModel):
    message: str

# issues a JWT when given a valid API key
@app.post("/token")
def get_token(api_key: str):
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    token = jwt.encode({"sub": api_key}, JWT_SECRET, algorithm=ALGORITHM)
    return {"token": token}

# validates the JWT on every protected route
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# Depends(verify_token) means this route requires a valid JWT
@app.post("/chat")
async def chat(request: ChatRequest, user=Depends(verify_token)):
    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=request.message
        )
        return {"response": response.text}
    except Exception as e:
        return {"error": str(e)}

handler = Mangum(app)