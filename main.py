from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel          # validates incoming request shape
from dotenv import load_dotenv          # reads .env file into environment
from google import genai
from mangum import Mangum               # translates Lambda events for FastAPI
from jose import jwt, JWTError          # JWT creation and validation
from sklearn.ensemble import IsolationForest
from sentence_transformers import SentenceTransformer
from datetime import datetime
import os
import redis
import numpy as np
import faiss
import boto3

load_dotenv()

# ---- app and client setup ----
app = FastAPI()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
JWT_SECRET = os.getenv("JWT_SECRET")
ALGORITHM = "HS256"                     # signing algorithm
MOCK_LLM = os.getenv("MOCK_LLM", "false").lower() == "true"
security = HTTPBearer()                 # tells FastAPI to expect a Bearer token in the header

# ---- DynamoDB — stores API keys instead of hardcoding them ----
dynamodb = boto3.resource("dynamodb", region_name="us-east-2")
api_keys_table = dynamodb.Table("llm-gateway-api-keys")

def validate_api_key(api_key: str) -> bool:
    # look up the key in DynamoDB — returns None if not found
    response = api_keys_table.get_item(Key={"api_key": api_key})
    item = response.get("Item")
    if not item:
        return False
    return item.get("is_active", False)  # reject if key is revoked

# ---- rate limiting — token bucket via Redis ----
redis_client = redis.Redis(host="localhost", port=6379, db=0)
RATE_LIMIT = 20     # max requests
WINDOW = 60         # per 60 seconds

def check_rate_limit(user_id: str):
    key = f"rate:{user_id}"
    current = redis_client.get(key)
    if current is None:
        # first request — start counter at 1, expires after WINDOW seconds
        redis_client.setex(key, WINDOW, 1)
    elif int(current) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again in a minute.")
    else:
        redis_client.incr(key)  # increment counter

# ---- anomaly detection — Isolation Forest on request features ----
request_logs = []
anomaly_detector = IsolationForest(contamination=0.1, random_state=42)

def log_and_check_anomaly(user_id: str, message: str):
    # extract features from this request
    features = {
        "hour": datetime.now().hour,            # time of day
        "prompt_length": len(message),           # how long the prompt is
        "user_id_hash": hash(user_id) % 1000    # which user (anonymized)
    }
    request_logs.append(features)

    # need at least 10 requests before training
    if len(request_logs) < 10:
        return

    # train model on all logs so far
    X = np.array([[r["hour"], r["prompt_length"], r["user_id_hash"]]
                   for r in request_logs])
    anomaly_detector.fit(X)

    # check if current request is anomalous (-1 = anomaly, 1 = normal)
    current = np.array([[features["hour"], features["prompt_length"], features["user_id_hash"]]])
    prediction = anomaly_detector.predict(current)
    if prediction[0] == -1:
        print(f"ANOMALY DETECTED: user={user_id}, hour={features['hour']}, prompt_length={features['prompt_length']}")

# ---- semantic cache — FAISS vector similarity search ----
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast embedding model
embedding_dim = 384     # vector size this model produces
faiss_index = faiss.IndexFlatL2(embedding_dim)
cache_store = []        # stores {"prompt": ..., "response": ...}

def check_semantic_cache(prompt: str):
    if len(cache_store) == 0:
        return None     # nothing cached yet

    # convert prompt to vector and search for closest match
    query_embedding = embedding_model.encode([prompt])
    distances, indices = faiss_index.search(query_embedding, k=1)

    # L2 distance below 0.5 means semantically similar
    if distances[0][0] < 0.5:
        print(f"CACHE HIT: '{prompt}' matched '{cache_store[indices[0][0]]['prompt']}'")
        return cache_store[indices[0][0]]["response"]

    return None     # no similar prompt found

def add_to_cache(prompt: str, response: str):
    # store embedding in FAISS and response in cache_store
    embedding = embedding_model.encode([prompt])
    faiss_index.add(embedding)
    cache_store.append({"prompt": prompt, "response": response})

# ---- request model ----
class ChatRequest(BaseModel):
    message: str    # request must include a "message" string field

# ---- routes ----
@app.post("/token")
def get_token(api_key: str):
    # validate API key against DynamoDB, return JWT if valid
    if not validate_api_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    token = jwt.encode({"sub": api_key}, JWT_SECRET, algorithm=ALGORITHM)
    return {"token": token}

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # decode and validate JWT on every protected request
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

@app.post("/chat")
async def chat(request: ChatRequest, user=Depends(verify_token)):
    check_rate_limit(user["sub"])           # enforce rate limit
    log_and_check_anomaly(user["sub"], request.message)  # flag anomalies

    # return cached response if semantically similar prompt exists
    cached = check_semantic_cache(request.message)
    if cached:
        return {"response": cached, "cached": True}

    if MOCK_LLM:
        response_text = f"Mock response to: {request.message}"
    else:
        try:
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=request.message
            )
            response_text = response.text
        except Exception as e:
            return {"error": str(e)}

    add_to_cache(request.message, response_text)
    return {"response": response_text, "cached": False}

# Lambda entry point — AWS looks for a variable named "handler"
handler = Mangum(app)