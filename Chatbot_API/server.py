from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from llm_api import TogetherAPIClient
import uvicorn
import uuid
from faiss_handler import initialize_faiss_indexs, store_conversation, retrieve_past_context

app = FastAPI()

# Initialize Together API Client
together_client = TogetherAPIClient('c500a3f01a29336d6918e96fdf59c4941d52ccc37cb1b4e46ee409adcba23ebb')




app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)

session_state = {}
@app.post("/start_session")
async def start_session():
    session_id = str(uuid.uuid4())
    
    session_state[session_id] = {
        "faiss_index": initialize_faiss_indexs(),
        "conversation_history": []
    }
    
    return JSONResponse(content={"session_id": session_id, "message": "Session Started"})

# Handle user requests and get the response from the LLM
@app.post("/get_response")
async def get_response(request: Request):
    data = await request.json()
    session_id = data['session_id']
    user_input = data['user_input']

    # Ensure session exists
    if session_id not in session_state:
        return JSONResponse(status_code=400, content={"error": "Invalid session_id"})

    # Retrieve past context based on user input using FAISS
    context_indexes = retrieve_past_context(session_id, user_input, session_state)
    
    if context_indexes[0][0] == -1:
        context = ""
    else:
        # Get valid conversation history based on FAISS indexes
        conversation_history = session_state[session_id]["conversation_history"]
        valid_indices = [i for i in context_indexes[0] if 0 <= i < len(conversation_history)]
        context = " ".join([conversation_history[i]['user_message'] for i in valid_indices])

    # Get response from Together API using the provided user input and context
    response = together_client.get_response(user_input, context)

    # Store the current user message and bot response in FAISS
    store_conversation(session_id, user_input, response, session_state)

    # Return the bot's response
    return {"response": response}

if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=5000, log_level="debug")
