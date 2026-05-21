from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel          # validates incoming request shape
from dotenv import load_dotenv          # reads .env file into os.environ
from google import genai
from mangum import Mangum               # translates Lambda events → ASGI for FastAPI
from jose import jwt, JWTError          # JWT creation and validation
from sklearn.ensemble import IsolationForest
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta, timezone
from typing import Optional
import os
import redis
import numpy as np
import faiss
import boto3
import hashlib                          # for cryptographic user ID hashing
import logging

load_dotenv()
logger = logging.getLogger(__name__)    # structured logging instead of bare print()

# ---- app and client setup ----
app = FastAPI()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
JWT_SECRET = os.getenv("JWT_SECRET")
ALGORITHM = "HS256"                     # HMAC-SHA256 — symmetric signing, same key signs and verifies
JWT_EXPIRY_MINUTES = int(os.getenv("JWT_EXPIRY_MINUTES", "30"))  # tokens expire after 30 min by default
MOCK_LLM = os.getenv("MOCK_LLM", "false").lower() == "true"      # skip real LLM calls during local dev
ADMIN_SECRET = os.getenv("ADMIN_SECRET")  # static secret for /admin/stats — separate from user JWTs
security = HTTPBearer()                 # tells FastAPI to expect a Bearer token in the Authorization header

# ---- DynamoDB — persistent storage for API keys and anomaly logs ----
# DynamoDB is used instead of hardcoding keys so we can revoke/rotate them without redeploying.
# It's also where we'll eventually persist anomaly records for offline analysis.
dynamodb = boto3.resource("dynamodb", region_name="us-east-2")
api_keys_table = dynamodb.Table("llm-gateway-api-keys")
request_log_table = dynamodb.Table("llm-gateway-request-logs")   # TODO: write anomalies here

def validate_api_key(api_key: str) -> bool:
    # DynamoDB get_item does a point lookup by primary key — O(1), no scan needed
    response = api_keys_table.get_item(Key={"api_key": api_key})
    item = response.get("Item")
    if not item:
        return False
    return item.get("is_active", False)  # keys can be soft-revoked without deletion

# ---- Redis — shared state for rate limiting across Lambda instances ----
# Lambda scales horizontally: multiple instances run simultaneously.
# If we stored counters in a Python variable, each instance would have its own counter,
# breaking rate limiting under concurrent traffic. Redis is the shared external store.
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")  # override to ElastiCache endpoint in prod
RATE_LIMIT = int(os.getenv("RATE_LIMIT", "20"))    # max requests per WINDOW seconds
WINDOW = 60                                         # sliding window length in seconds

def get_redis():
    # Try to connect once at startup. If Redis is unavailable, we degrade gracefully
    # rather than crashing the whole service — rate limiting becomes best-effort.
    try:
        r = redis.Redis(host=REDIS_HOST, port=6379, db=0, socket_connect_timeout=1)
        r.ping()
        return r
    except Exception:
        logger.warning("Redis unavailable — rate limiting and stats will degrade gracefully")
        return None

redis_client = get_redis()

def check_rate_limit(user_id: str):
    if redis_client is None:
        return  # fail open — better to serve than to block all traffic when Redis is down

    key = f"rate:{user_id}"
    try:
        current = redis_client.get(key)
        if current is None:
            # First request in this window: set counter to 1, auto-expire after WINDOW seconds.
            # SETEX is atomic — no race condition between SET and EXPIRE.
            redis_client.setex(key, WINDOW, 1)
        elif int(current) >= RATE_LIMIT:
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again in a minute.")
        else:
            redis_client.incr(key)   # INCR is atomic, safe under concurrent Lambda instances
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Rate limit check failed: {e}")  # Redis hiccup — let request through

def check_token_rate_limit(ip: str):
    # Separate rate limit for /token to prevent brute-forcing API keys.
    # 10 attempts/min per IP is generous for legitimate use but blocks automated scanners.
    if redis_client is None:
        return
    key = f"token_rate:{ip}"
    try:
        current = redis_client.get(key)
        if current is None:
            redis_client.setex(key, 60, 1)
        elif int(current) >= 10:
            raise HTTPException(status_code=429, detail="Too many token requests. Slow down.")
        else:
            redis_client.incr(key)
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Token rate limit check failed: {e}")

# ---- anomaly detection — Isolation Forest on request features ----
# Isolation Forest works without labeled training data: it learns what "normal" looks like
# by randomly partitioning the feature space. Anomalies are isolated in fewer splits
# because they're statistically rare and far from the dense normal region.
request_logs = []
anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
# contamination=0.1 means we expect ~10% of requests to be anomalous.
# random_state=42 makes results reproducible across retrains.

RETRAIN_EVERY = 50   # retrain periodically rather than on every single request.
                     # Retraining is O(n) — doing it every call makes the gateway
                     # get slower as logs accumulate.
_request_count_since_retrain = 0

def _safe_hash(user_id: str) -> int:
    # Python's built-in hash() is NOT cryptographic and changes between processes.
    # SHA-256 gives a stable, non-reversible numeric ID for the anomaly feature vector.
    return int(hashlib.sha256(user_id.encode()).hexdigest(), 16) % 10000

def log_and_check_anomaly(user_id: str, message: str):
    global _request_count_since_retrain
    # Features chosen because they capture different axes of "unusual" behavior:
    #   hour         — requests at 3am from a business account are suspicious
    #   prompt_length — extremely long prompts may be prompt-injection attempts
    #   user_id_hash  — sudden new user flooding the system is suspicious
    features = {
        "hour": datetime.now().hour,
        "prompt_length": len(message),
        "user_id_hash": _safe_hash(user_id),
    }
    request_logs.append(features)
    _request_count_since_retrain += 1

    if len(request_logs) < 10:
        return  # need a minimum sample before the model has any signal

    # Only retrain every RETRAIN_EVERY requests to keep overhead constant
    if _request_count_since_retrain >= RETRAIN_EVERY:
        X = np.array([[r["hour"], r["prompt_length"], r["user_id_hash"]] for r in request_logs])
        anomaly_detector.fit(X)
        _request_count_since_retrain = 0
        logger.info(f"Anomaly detector retrained on {len(request_logs)} samples")

    try:
        current = np.array([[features["hour"], features["prompt_length"], features["user_id_hash"]]])
        prediction = anomaly_detector.predict(current)
        # predict() returns 1 for normal, -1 for anomaly
        if prediction[0] == -1:
            logger.warning(
                f"ANOMALY DETECTED: user={user_id}, "
                f"hour={features['hour']}, prompt_length={features['prompt_length']}"
            )
            # TODO: persist to DynamoDB request_log_table for offline analysis and alerting
    except Exception as e:
        logger.warning(f"Anomaly check failed: {e}")

# ---- semantic cache — FAISS vector similarity search ----
# Why semantic instead of exact-match? Natural language has infinite variation.
# "What is ML?" and "Explain machine learning" are the same question but would
# never match a string cache. Embedding-based similarity catches semantic
# equivalence regardless of phrasing.
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # small (80MB), fast, good quality
embedding_dim = 384       # vector size this model produces — must match IndexFlatL2 dimension
faiss_index = faiss.IndexFlatL2(embedding_dim)  # exact L2 search — fine for small caches;
                                                # switch to IndexIVFFlat for 10k+ entries
cache_store = []          # parallel list to faiss_index — stores {"prompt": ..., "response": ...}
                          # indices must stay in sync: cache_store[i] matches faiss_index vector i
CACHE_THRESHOLD = 0.5     # L2 distance below this = semantically similar enough to serve cached

def check_semantic_cache(prompt: str, embedding: np.ndarray) -> Optional[str]:
    # Takes a pre-computed embedding to avoid recomputing on cache miss.
    # faiss_index.search() returns (distances, indices) arrays of shape (n_queries, k)
    if len(cache_store) == 0:
        return None
    distances, indices = faiss_index.search(embedding, k=1)  # find the single closest vector
    if distances[0][0] < CACHE_THRESHOLD:
        matched_prompt = cache_store[indices[0][0]]["prompt"]
        logger.info(f"CACHE HIT: '{prompt[:40]}' matched '{matched_prompt[:40]}'")
        return cache_store[indices[0][0]]["response"]
    return None

def add_to_cache(prompt: str, response: str, embedding: np.ndarray):
    # faiss_index and cache_store must be updated together to stay in sync.
    # faiss_index.add() appends to the index; cache_store.append() appends to the list.
    # They share the same implicit integer index.
    faiss_index.add(embedding)
    cache_store.append({"prompt": prompt, "response": response})

# ---- request counters in Redis (accurate across multiple Lambda instances) ----
# In-memory globals like `total_requests += 1` break under horizontal scaling:
# each Lambda instance increments its own copy. Redis INCR is atomic and shared.
def _redis_incr_stat(key: str):
    if redis_client is None:
        return
    try:
        redis_client.incr(key)
    except Exception:
        pass  # stat tracking is best-effort — never block a request over it

def get_stats_from_redis():
    if redis_client is None:
        return {"total_requests": "unavailable", "cache_hits": "unavailable", "cache_hit_rate": "unavailable"}
    try:
        total = int(redis_client.get("stat:total_requests") or 0)
        hits = int(redis_client.get("stat:cache_hits") or 0)
        rate = (hits / total * 100) if total > 0 else 0
        return {"total_requests": total, "cache_hits": hits, "cache_hit_rate": f"{rate:.1f}%"}
    except Exception:
        return {"total_requests": "unavailable", "cache_hits": "unavailable", "cache_hit_rate": "unavailable"}

# ---- request model ----
class ChatRequest(BaseModel):
    message: str    # Pydantic validates that "message" is present and is a string.
                    # Returns 422 automatically if the field is missing or wrong type.

# ---- JWT helpers ----
def create_jwt(api_key: str) -> str:
    # Standard JWT claims:
    #   sub  — subject, who this token represents
    #   iat  — issued-at timestamp
    #   exp  — expiration timestamp (python-jose enforces this automatically on decode)
    # Using timezone-aware UTC datetimes avoids subtle bugs with naive datetime comparisons.
    payload = {
        "sub": api_key,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRY_MINUTES),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # jwt.decode() validates the signature AND the exp claim in one call.
    # JWTs are self-verifying — no database lookup needed on every request,
    # keeping the auth layer stateless and horizontally scalable.
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# ---- routes ----
@app.post("/token")
def get_token(api_key: str, request: Request):
    # Rate-limit by client IP before touching DynamoDB — cheap check first.
    client_ip = request.client.host if request.client else "unknown"
    check_token_rate_limit(client_ip)

    if not validate_api_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    token = create_jwt(api_key)
    return {"token": token, "expires_in": JWT_EXPIRY_MINUTES * 60}  # expires_in in seconds

@app.get("/admin/stats")
def get_stats(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Admin endpoint uses a static secret rather than user JWTs so operators
    # can check stats even when the user JWT pool is empty or rotated.
    if credentials.credentials != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")
    stats = get_stats_from_redis()
    stats["cache_size"] = len(cache_store)         # how many prompts are cached
    stats["anomaly_samples"] = len(request_logs)   # how many samples the model has trained on
    return stats

@app.post("/chat")
async def chat(request: ChatRequest, user=Depends(verify_token)):
    user_id = user["sub"]   # the API key that was exchanged for this JWT

    # Pipeline: rate limit → anomaly check → cache lookup → LLM call
    # Order matters: rate limit first (cheapest), LLM call last (most expensive)
    check_rate_limit(user_id)
    log_and_check_anomaly(user_id, request.message)

    # Compute embedding once here and pass it to both cache functions.
    # embedding_model.encode() is the most expensive local operation (~10ms on CPU)
    # — computing it twice for every cache miss would be wasteful.
    embedding = embedding_model.encode([request.message])

    cached = check_semantic_cache(request.message, embedding)
    if cached:
        _redis_incr_stat("stat:total_requests")
        _redis_incr_stat("stat:cache_hits")
        return {"response": cached, "cached": True}

    _redis_incr_stat("stat:total_requests")

    if MOCK_LLM:
        # Mock path also writes to cache so cache behavior is testable locally
        response_text = f"Mock response to: {request.message}"
    else:
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",   # corrected model name (gemini-3 does not exist)
                contents=request.message,
            )
            response_text = response.text
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise HTTPException(status_code=502, detail="LLM provider error")
            # 502 Bad Gateway is semantically correct: we received a bad response
            # from an upstream server (Gemini). Better than returning 200 with an error body.

    add_to_cache(request.message, response_text, embedding)
    return {"response": response_text, "cached": False}

# Lambda entry point — AWS Lambda looks for a callable named "handler" in the module.
# Mangum wraps the FastAPI ASGI app and translates Lambda's event/context format
# into the ASGI lifespan/request/response protocol FastAPI understands.
handler = Mangum(app)