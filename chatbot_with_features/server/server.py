from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# from model_pipeline import TogetherAPIClient
from llm_api import BenAssistant
import uvicorn
from chroma_handler import VectorStore  


app = FastAPI()
together_client = BenAssistant('c500a3f01a29336d6918e96fdf59c4941d52ccc37cb1b4e46ee409adcba23ebb')
vector_store = VectorStore(collection_name="chatbot_conversations")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)


class UserInputRequest(BaseModel):
    user_input: str


#Get Chatbot Response.
@app.post("/get_response")
async def get_response(request: UserInputRequest):
    user_input =  request.user_input
    # context_results = vector_store.search_context(user_input) 
    response = together_client.get_response(user_input)
    together_client.messages.append({"User": f"{user_input}","Bot": f"{response}" })
    # vector_store.populate_vectors(user_input, response)
    return {"response": response}


#Get next possible questions based on user query.
@app.post('/intellisense_questions')
async def intellisense_questions(request: UserInputRequest):
    user_input =  request.user_input
    question_list = together_client.get_question_recommendation(user_input)
    return {"response":question_list}
    
    
if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=5000, log_level="debug")
    """http://localhost:5000/docs"""