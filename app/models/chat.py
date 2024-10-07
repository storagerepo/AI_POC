from pydantic import BaseModel
from typing import List, Union, Dict, Any, Optional

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[Union[Message, Dict[str, Any]]] = []
    document_ids: Optional[List[str]] = None
    
class SimilarChat(BaseModel):
    id: int
    user_message: str
    ai_response: str
    similarity: float

class ChatResponse(BaseModel):
    response: str
    history: List[Union[Message, Dict[str, Any]]]
    similar_chats: Optional[List[SimilarChat]] = None
