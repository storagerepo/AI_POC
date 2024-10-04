from fastapi import FastAPI, Request,APIRouter
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.api.llm_api import TogetherAPIClient
import uvicorn
from app.api.chroma_handler import VectorStore 
from pydantic import BaseModel

vector_store = VectorStore(collection_name="chatbot_conversations")
together_client = TogetherAPIClient('c500a3f01a29336d6918e96fdf59c4941d52ccc37cb1b4e46ee409adcba23ebb')

class InputModel(BaseModel):
   user_input:str

class ChatBotResponse(BaseModel):
    response: str

router = APIRouter()

@router.post("/getResponse", response_model=ChatBotResponse)
async def getResponse(input_data: InputModel):
    print(input_data)
    user_input = input_data.user_input
    context_results = vector_store.search_context(user_input) 
    response = together_client.get_response(user_input, context_results)
    # vector_store.populate_vectors(user_input, response)
    print("dsds",response) 
    return {"response": response}
    