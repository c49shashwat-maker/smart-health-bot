from __future__ import annotations
import os
from typing import Literal
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

app = FastAPI(title="Smart Health Bot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = (
    "You are Smart Health Bot, a friendly, objective general health information assistant. "
    "Provide clear, evidence-based educational information on wellness, nutrition, fitness, "
    "sleep, and general medical concepts. Never diagnose conditions, prescribe medications, "
    "or replace professional medical care. If a user describes severe or emergency symptoms "
    "(such as chest pain, trouble breathing, severe bleeding, or stroke signs), advise them "
    "to seek immediate emergency medical care. Keep answers well-structured with short "
    "paragraphs and bullet points."
)

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: list[Message] = Field(default_factory=list)
    provider: Literal["gemini", "groq"] = "gemini"

class ChatResponse(BaseModel):
    reply: str
    provider: str

def _call_gemini(message: str, history: list[Message]) -> str:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY is not set in backend/.env")
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=api_key)
        contents = []
        for msg in history:
            role = "user" if msg.role == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part(text=msg.content)]))
        contents.append(types.Content(role="user", parts=[types.Part(text=message)]))
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.7,
                max_output_tokens=1024,
            ),
        )
        return response.text or "Sorry, I could not generate a response."
    except HTTPException:
        raise
    except Exception as exc:
        err = str(exc)
        if "API_KEY_INVALID" in err or "401" in err:
            raise HTTPException(status_code=401, detail="Invalid Gemini API key.")
        if "429" in err or "quota" in err.lower() or "rate" in err.lower():
            raise HTTPException(status_code=429, detail="Gemini rate limit reached. Wait and retry.")
        raise HTTPException(status_code=500, detail=f"Gemini error: {err}")

def _call_groq(message: str, history: list[Message]) -> str:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is not set in backend/.env")
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in history:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": message})
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
        )
        return completion.choices[0].message.content or "Sorry, I could not generate a response."
    except HTTPException:
        raise
    except Exception as exc:
        err = str(exc)
        if "invalid_api_key" in err or "401" in err:
            raise HTTPException(status_code=401, detail="Invalid Groq API key.")
        if "429" in err or "rate_limit" in err.lower():
            raise HTTPException(status_code=429, detail="Groq rate limit reached. Wait and retry.")
        raise HTTPException(status_code=500, detail=f"Groq error: {err}")

def _key_valid(val: str) -> bool:
    return bool(val.strip()) and val.strip() not in ("your_gemini_api_key_here", "your_groq_api_key_here")

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "gemini_configured": _key_valid(os.getenv("GEMINI_API_KEY", "")),
        "groq_configured": _key_valid(os.getenv("GROQ_API_KEY", "")),
    }

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if req.provider == "gemini":
        reply = _call_gemini(req.message, req.history)
    else:
        reply = _call_groq(req.message, req.history)
    return ChatResponse(reply=reply, provider=req.provider)
