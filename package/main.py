from fastapi import FastAPI
from pydantic import BaseModel # validates incoming request shape
from dotenv import load_dotenv # reads .env file into environment
from google import genai
from mangum import Mangum      # translates Lambda events for FastAPI
import os

load_dotenv()

app = FastAPI()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class ChatRequest(BaseModel):
    # request must include a "message" string field
    message: str  

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=request.message
        )
        return {"response": response.text}
    except Exception as e:
        return {"error": str(e)}

# Lambda entry point — AWS looks for a variable named "handler"
handler = Mangum(app)