from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class ChatRequest(BaseModel):
    company_name: str
    message: str
    session_id: Optional[int] = None
    session_name: Optional[str] = None
    # Optional list of document IDs to restrict retrieval to
    specific_documents: Optional[List[int]] = None

class ChatResponse(BaseModel):
    session_id: int
    response: str
    context_documents: List[dict]
    processing_time_ms: Optional[float] = None

class SessionCreate(BaseModel):
    company_name: str
    session_name: str

class SessionResponse(BaseModel):
    id: int
    name: str
    created_at: datetime
    updated_at: datetime
    message_count: int

class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    timestamp: datetime
    context_documents: Optional[List[int]] = None

class QuickPrompt(BaseModel):
    company_name: str
    prompt_type: str  # "summarize", "analyze_risks", "key_terms", "obligations", "deadlines"
    session_id: Optional[int] = None
    specific_documents: Optional[List[int]] = None  # Specific document IDs to focus on