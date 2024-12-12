from fastapi import FastAPI, Response
from pydantic import BaseModel
from typing import List, Optional, Dict
from chatting.chatting import (
    Agent, 
    chat
)
import requests

app = FastAPI()

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

@app.get('/ask')
def ask(prompt :str):
    res = requests.post('http://ollama:11434/api/generate', json={
        "prompt": prompt,
        "stream" : False,
        "model" : "llama3.2:1b"
    })

    return Response(content=res.text, media_type="application/json")

@app.post("/chat")
def start_chat(chat_request: ChatRequestModel):
    dialog = chat(
                    agents=chat_request.agents,
                    subject=chat_request.subject,
                    use_memory=chat_request.use_memory,
                    use_location=chat_request.location
                )

    return {"dialog": dialog}
