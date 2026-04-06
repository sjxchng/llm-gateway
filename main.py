from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from mangum import Mangum
from jose import jwt, JWTError  # JWT creation and validation
import os

import redis
import time

from sklearn.ensemble import IsolationForest
import numpy as np
import json
from datetime import datetime

from sentence_transformers import SentenceTransformer
import faiss


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


# mock
MOCK_LLM = os.getenv("MOCK_LLM", "false").lower() == "true"

@app.post("/chat")
async def chat(request: ChatRequest, user=Depends(verify_token)):
    check_rate_limit(user["sub"])
    log_and_check_anomaly(user["sub"], request.message)
    
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
    
    
    
    
# connect to Redis for rate limiting
redis_client = redis.Redis(host="localhost", port=6379, db=0)

RATE_LIMIT = 20      # max requests
WINDOW = 60          # per 60 seconds

def check_rate_limit(user_id: str):
    key = f"rate:{user_id}"
    current = redis_client.get(key)
    if current is None:
        # first request — set counter to 1 with 60 second expiry
        redis_client.setex(key, WINDOW, 1)
    elif int(current) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again in a minute.")
    else:
        redis_client.incr(key)
        
# in-memory request log (in production this would be DynamoDB)
request_logs = []

# anomaly detector — trained on request features
anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
is_trained = False  # tracks whether the model has been trained yet

def log_and_check_anomaly(user_id: str, message: str):
    global is_trained
    
    # extract features from this request
    features = {
        "hour": datetime.now().hour,          # time of day
        "prompt_length": len(message),         # how long the prompt is
        "user_id_hash": hash(user_id) % 1000  # which user (anonymized)
    }
    request_logs.append(features)
    
    # only run anomaly detection once we have enough data to train on
    if len(request_logs) < 10:
        return
    
    # convert logs to a matrix and train/retrain the model
    X = np.array([[r["hour"], r["prompt_length"], r["user_id_hash"]] 
                   for r in request_logs])
    anomaly_detector.fit(X)
    is_trained = True
    
    # check if the current request is anomalous (-1 = anomaly, 1 = normal)
    current = np.array([[features["hour"], features["prompt_length"], features["user_id_hash"]]])
    prediction = anomaly_detector.predict(current)
    
    if prediction[0] == -1:
        print(f"ANOMALY DETECTED: user={user_id}, hour={features['hour']}, prompt_length={features['prompt_length']}")
        
# semantic cache setup
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast embedding model
embedding_dim = 384  # dimension of the vectors this model produces

# FAISS index stores all the embedding vectors
faiss_index = faiss.IndexFlatL2(embedding_dim)

# stores the actual cached responses mapped to their index position
cache_store = []  # list of {"prompt": ..., "response": ...}

SIMILARITY_THRESHOLD = 0.92

def check_semantic_cache(prompt: str):
    if len(cache_store) == 0:
        return None  # cache is empty, nothing to check
    
    # convert prompt to embedding vector
    query_embedding = embedding_model.encode([prompt])
    
    # search FAISS for the closest cached prompt
    distances, indices = faiss_index.search(query_embedding, k=1)
    
    closest_distance = distances[0][0]
    closest_index = indices[0][0]
    
    # L2 distance — lower means more similar, 0 means identical
    # we use 0.5 as threshold (roughly equivalent to 0.92 cosine similarity)
    if closest_distance < 0.5:
        print(f"CACHE HIT: '{prompt}' matched cached prompt '{cache_store[closest_index]['prompt']}'")
        return cache_store[closest_index]["response"]
    
    return None  # no similar prompt found

def add_to_cache(prompt: str, response: str):
    embedding = embedding_model.encode([prompt])
    faiss_index.add(embedding)
    cache_store.append({"prompt": prompt, "response": response})
    
        
handler = Mangum(app)