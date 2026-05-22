from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from agent import agent_service
from schemas import AgentResponse

app = FastAPI()

class GenerateRequest(BaseModel):
    message: str

@app.post(
    "/generate",
    response_model=AgentResponse
)
async def generate(req: GenerateRequest):
    return await agent_service.generate_answer(
        user_input=req.message
    )

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="localhost",
        port=8000
    )