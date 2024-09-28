import streamlit as st
import requests
import time 

GET_LLAMA_RESPONSE_URL = "http://localhost:5000/get_response"

def get_llama_response(user_input):
    data = {
        "user_input": user_input
    }
    response = requests.post(GET_LLAMA_RESPONSE_URL, json=data)

    if response.status_code == 200:
        return response.json().get("response", "No response available.")
    else:
        st.error(f"Error: Unable to get response, {response.status_code}")
        return "Error: Unable to get response."

#Just to look like streaming
def typewriter_effect(text):
    response_container = st.empty()  
    displayed_text = ""
    for char in text:
        displayed_text += char
        response_container.markdown(displayed_text)
        time.sleep(0.01)  

# Frontend setup using Streamlit
st.title("Ask Ben")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("What is up?"):

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = get_llama_response(user_input)
        if response:
            typewriter_effect(response)

    if response:
        st.session_state.messages.append({"role": "assistant", "content": response})

st.sidebar.header("Instructions")
st.sidebar.write("Type your message in the input box and press Enter to chat with BEN!")

