from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from llm_api import TogetherAPIClient
import uvicorn
from chroma_handler import VectorStore  

app = FastAPI()

together_client = TogetherAPIClient('c500a3f01a29336d6918e96fdf59c4941d52ccc37cb1b4e46ee409adcba23ebb')

vector_store = VectorStore(collection_name="chatbot_conversations")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)


@app.post("/get_response")
async def get_response(request: Request):
    data = await request.json()
    user_input = data['user_input']

    context_results = vector_store.search_context(user_input,n_results=5) 
    context = " ".join(result['document'] for result in context_results) 
    response = together_client.get_response(user_input, context)
    vector_store.populate_vectors(user_input, response)

    return {"response": response}

if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=5000, log_level="debug")