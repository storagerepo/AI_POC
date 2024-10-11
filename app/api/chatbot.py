from fastapi import FastAPI, Request,APIRouter
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from api.llm_api import TogetherAPIClient
from api.chroma_handler import VectorStore 
from pydantic import BaseModel
import json

vector_store = VectorStore(collection_name="chatbot_conversations")
together_client = TogetherAPIClient('c500a3f01a29336d6918e96fdf59c4941d52ccc37cb1b4e46ee409adcba23ebb')

class InputModel(BaseModel):
   user_input:str

class ChatBotResponse(BaseModel):
    response: str

router = APIRouter()

@router.post("/getResponse", response_model=ChatBotResponse)
async def getResponse(input_data: InputModel):
    user_input = input_data.user_input
    print("Input Data: ",user_input)
    context_results = vector_store.search_context(user_input) 
    
    intent = together_client.classify_intent(query=user_input)
    print("Intent: ",intent)

    if intent['intent'] == "SEARCH_INTENT":
        entities = together_client.extract_entities(user_input)
        print("Extracted Entities:",json.dumps(entities, indent=2))
        response = "Here are the Properties you are looking for..."
        vector_store.populate_vectors(user_input, response)
        print("Total Token count: ",together_client.token_count)
        return {"response": response}
        
    else:
        response = together_client.get_response(user_input,context_results)
        print("Total Token count: ",together_client.token_count)
        vector_store.populate_vectors(user_input, response)
        return {"response": response}


    