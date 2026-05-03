from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message/query")
    thread_id: str = Field(..., min_length=1, description="Conversation/session ID")


class ChatResponse(BaseModel):
    answer: str
    status: str
    thread_id: str
    safety_status: str | None = None
    waiting_for_hitl: bool = False


class ApproveRequest(BaseModel):
    thread_id: str = Field(..., description="The session ID to resume")
    feedback: str | None = Field(None, description="Optional feedback for refinement")