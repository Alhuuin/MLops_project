from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict
from chatting.chatting import (
    Agent, 
    chat
)
import requests
from decouple import config
import random

app = FastAPI()

VALID_TOKEN = config("VALID_TOKEN")

base_model = "llama3.2:1b"
next_model = "llama3.2:1b"
p = 0.8

class AgentModel(BaseModel):
    agent_id: str
    name: str
    user_input: str
    gender: str
    opinions: Dict[str, str] = {}


class ChatRequestModel(BaseModel):
    agents: List[AgentModel]
    subject: str
    location: Optional[str] = None
    use_memory: bool = True

def verify_token(authorization: str = Header(...)):
    if authorization != VALID_TOKEN:
        raise HTTPException(status_code=400, detail="Invalid or missing token.")

@app.post("/update-model")
def update_model(new_version: str, authorization: str = Depends(verify_token)):
    global next_model
    next_model = new_version
    return {"message": f"Next model updated to {new_version}"}

@app.post("/accept-next-model")
def accept_next_model(authorization: str = Depends(verify_token)):
    global base_model, next_model
    base_model = next_model
    return {"message": "Next model accepted as the current model"}


@app.post("/chat")
def start_chat(chat_request: ChatRequestModel, authorization: str = Depends(verify_token)):
    chosen_model = base_model if random.random() < p  else next_model
    print(f"model chosen for this request: {chosen_model}")
    dialog, updated_agents = chat(
                    agents=chat_request.agents,
                    subject=chat_request.subject,
                    model = chosen_model,
                    use_memory=chat_request.use_memory,
                    use_location=chat_request.location
                )

    return {"dialog": dialog, "updated_agents": updated_agents}
