from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai

import os

load_dotenv()

app = FastAPI()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=request.message
    )
    return {"response": response.text}