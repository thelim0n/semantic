from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import generate_answer
import uvicorn

app = FastAPI(title="Investment Advisor API")

class GenerateRequest(BaseModel):
    message: str

class GenerateResponse(BaseModel):
    answer: str | None = None

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    result = generate_answer(req.message)
    return result

if __name__ == "__main__":
    uvicorn.run(
        "app:app", 
        host="localhost",
        reload=True
    )