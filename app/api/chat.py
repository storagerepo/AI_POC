from fastapi import APIRouter, HTTPException
from app.models.chat import ChatRequest, ChatResponse, Message, SimilarChat
from app.models.documents import DocumentResponse, DocumentUpload
from app.services.llm_service import get_model_response
from app.services.milvus_service import milvus_service
from app.utils.helpers import get_embedding

from fastapi import APIRouter, HTTPException, UploadFile, File

router = APIRouter()



    
@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    messages = [msg.dict() for msg in request.history]
    messages.append(
        {
            "role":"user",
            "content": request.message,
        }
    )
    try:  
        
        user_embedding = get_embedding(request.message)
        
        
        context = milvus_service.get_conversation_context(user_embedding)
        
        context_messages = []
        for conv in context:
            context_messages.append({"role": "user", "content": conv["user_message"]})
            context_messages.append({"role": "assistant", "content": conv["ai_response"]})
        
        full_context = context_messages + messages
        
        ai_response = get_model_response(messages=full_context)

        chat_id = milvus_service.insert_chat(request.message, ai_response, user_embedding)
        
        similar_chats = milvus_service.search_similar_chats(user_embedding, limit=3)
        
        similar_chat_objects = [
            SimilarChat(
                id=chat['id'],
                user_message=chat['user_message'],
                ai_response=chat['ai_response'],
                similarity=1 - chat['distance']  # Convert distance to similarity
            )
            for chat in similar_chats if chat['id'] != chat_id  # Exclude the current chat
        ]
        
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(errors=error))
    
    messages.append(
        {
            "role":"assistant",
            "content": ai_response
        }
    )
    
    updated_history = [Message(**msg) for msg in messages]
    
    return ChatResponse(response=ai_response, history=updated_history, similar_chats=similar_chat_objects)

@router.get("/chat/{chat_id}", response_model=SimilarChat)
async def get_chat(chat_id: int):
    chat = milvus_service.get_chat_by_id(chat_id)
    if chat:
        return SimilarChat(
            id=chat['id'],
            user_message=chat['user_message'],
            ai_response=chat['ai_response'],
            similarity=1.0  # Full similarity for exact match
        )
    raise HTTPException(status_code=404, detail="Chat not found")