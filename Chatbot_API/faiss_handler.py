import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

index_file_path = "faissDB/faiss_index_file.index"
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embedding(text):
    return embedding_model.encode(text).astype(np.float32)

def initialize_faiss_indexs():
    dimension = 384  
    faiss_index = faiss.IndexFlatL2(dimension)
    print("New Faiss index created.")
    return faiss_index

def save_faiss_index(faiss_index):
    try:
        faiss.write_index(faiss_index, index_file_path)
        print("Faiss index saved to disk successfully.")
    except Exception as e:
        print(f"Error saving Faiss index to disk: {e}")

def retrieve_past_context(session_id, user_message, session_state):
    faiss_index = session_state[session_id]['faiss_index']
    vector = generate_embedding(user_message).astype(np.float32)
    D, I = faiss_index.search(np.array([vector]), k=3)  
    return I  

def store_conversation(session_id, user_message, bot_response, session_state):
    faiss_index = session_state[session_id]['faiss_index']
    conversation_history = session_state[session_id]['conversation_history']

    vector = generate_embedding(user_message).astype(np.float32)  
    faiss_index.add(np.array([vector]))  
    conversation_history.append({"user_message": user_message, "bot_response": bot_response})
    save_faiss_index(faiss_index)  
