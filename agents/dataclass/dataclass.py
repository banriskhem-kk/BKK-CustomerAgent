from pydantic import BaseModel
from typing import Dict, Any, List


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ToolCall(BaseModel):
    tool: str
    input: Dict[str, str]
    result: Dict[str, Any]


class ChatAgentResponse(BaseModel):
    response: str
    agent: str
    tool_calls: List[ToolCall]
    handover: str
