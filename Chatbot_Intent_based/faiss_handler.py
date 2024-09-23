import faiss
import numpy as np
import streamlit as st
from embedding import generate_embedding

index_file_path = "faissDB/faiss_index_file.index"

def initialize_faiss_indexs():
    if "faiss_index" not in st.session_state:
         dimension = 384  
         st.session_state.faiss_index = faiss.IndexFlatL2(dimension)
         print("New Faiss index created.")
         
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
        
    if "messages" not in st.session_state:
        st.session_state.messages = []

def save_faiss_index():
    try:
        faiss.write_index(st.session_state.faiss_index, index_file_path)
        print("Faiss index saved to disk successfully.")
    except Exception as e:
        print(f"Error saving Faiss index to disk: {e}")


def retrieve_past_context(user_message):
    print(user_message)
    vector = generate_embedding(user_message).astype(np.float32)
    D, I = st.session_state.faiss_index.search(np.array([vector]), k=3)  
    return I  


def store_conversation(user_message, bot_response):
    vector = generate_embedding(user_message).astype(np.float32)  
    st.session_state.faiss_index.add(np.array([vector]))  
    st.session_state.conversation_history.append({"user_message": user_message, "bot_response": bot_response})
    print(st.session_state.conversation_history)
    print("Storing vector for message:", user_message)
    print("Vector:", vector)
    save_faiss_index()  
    

def save_faiss_index():
    try:
        faiss.write_index(st.session_state.faiss_index, index_file_path)
        print("Faiss index saved to disk successfully.")
    except Exception as e:
        print(f"Error saving Faiss index to disk: {e}")
