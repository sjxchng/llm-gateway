# LLM API Gateway

A production-style LLM API gateway built with FastAPI and deployed on AWS Lambda. Sits between any application and an LLM provider, handling authentication, rate limiting, anomaly detection, and semantic caching.

## Architecture

```
Client → API Gateway → Lambda → Auth → Rate Limit → Anomaly Detection → Semantic Cache → LLM
```

Every request passes through four layers before reaching the LLM:

1. **JWT Authentication** — callers must present a valid signed token
2. **Rate Limiting** — token bucket enforces 20 requests/minute per user via Redis
3. **Anomaly Detection** — Isolation Forest flags unusual request patterns
4. **Semantic Cache** — FAISS vector search returns cached responses for similar prompts

## Stack

| Component | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| Deployment | AWS Lambda + API Gateway |
| Lambda Adapter | Mangum |
| Auth | JWT (python-jose) |
| API Key Storage | AWS DynamoDB |
| Rate Limiting | Redis (AWS ElastiCache) |
| Anomaly Detection | scikit-learn (Isolation Forest) |
| Semantic Cache | FAISS + sentence-transformers |
| LLM | Google Gemini |

## Endpoints

| Method | Path | Auth Required | Description |
|---|---|---|---|
| POST | `/token` | No | Exchange API key for JWT |
| POST | `/chat` | Yes | Send message to LLM |
| GET | `/admin/stats` | No | Cache hit rate and request counts |

## Setup

### Prerequisites

- Python 3.11+
- Docker (for Lambda deployment builds)
- AWS CLI configured (`aws configure`)
- Redis running locally (`docker run -d -p 6379:6379 redis`)

### Local Development

```bash
# Clone the repo
git clone https://github.com/sjxchng/llm-gateway
cd llm-gateway

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your keys

# Run locally
uvicorn main:app --reload
```

### Environment Variables

```
GEMINI_API_KEY=your_gemini_key
JWT_SECRET=your_secret_key
MOCK_LLM=false
```

### Testing the API

```bash
# 1. Get a token
curl -X POST "http://127.0.0.1:8000/token?api_key=your-api-key"

# 2. Send a chat message
curl -X POST http://127.0.0.1:8000/chat \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "what is machine learning?"}'

# 3. Check cache stats
curl http://127.0.0.1:8000/admin/stats
```

## Benchmark Results

Semantic cache hit rate measured over 50 requests with semantically similar prompts:

| Metric | Value |
|---|---|
| Total requests | 50 |
| Cache hits | ~X |
| Cache hit rate | ~X% |

> Run the benchmark yourself: fire 50 requests with variations of the same questions and check `/admin/stats`

## Deployment

Dependencies are built inside a Docker container matching Lambda's x86_64 runtime to avoid binary incompatibility issues:

```bash
# Build Linux-compatible dependencies
docker run --rm \
  -v $(pwd)/package:/var/task \
  public.ecr.aws/lambda/python:3.13 \
  pip install -r requirements.txt -t /var/task

# Package and upload
cp main.py package/
cd package && zip -r ../lambda.zip . && cd ..
aws s3 cp lambda.zip s3://llm-gateway-deployment/lambda.zip
```

## Design Decisions

**Why Redis for rate limiting?** Lambda scales horizontally — multiple instances can run simultaneously. Storing counts in a Python variable would give each instance its own counter, breaking rate limiting under concurrent traffic. Redis is a shared external store all instances read from.

**Why semantic cache instead of exact-match?** Natural language has infinite variation. "What is ML?" and "Explain machine learning" are the same question but would never match an exact string cache. Embedding-based similarity catches semantic equivalence regardless of phrasing.

**Why Isolation Forest?** Anomaly detection requires no labeled training data — you don't need examples of "bad" requests to train it. It learns what normal looks like and flags statistical outliers.

**Why JWT over sessions?** Sessions require a database lookup on every request. JWTs are self-verifying — the server checks the cryptographic signature without any database call, making the system stateless and horizontally scalable.

## Known Limitations & Future Improvements

- Redis runs locally — production would use AWS ElastiCache
- FAISS index is in-memory — would persist to S3 on shutdown
- JWT has no expiration — should add `exp` claim
- Anomaly detector retrains on every request — should train periodically
- No rate limiting on `/token` endpoint — vulnerable to brute force
