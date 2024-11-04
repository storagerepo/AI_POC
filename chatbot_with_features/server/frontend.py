import streamlit as st
import requests
import time

# API Endpoints
GET_LLAMA_RESPONSE_URL = "http://localhost:5000/get_response"
GET_QUESTION_RECOMMENDATIONS_URL = "http://localhost:5000/intellisense_questions"

# API call functions
def get_llama_response(user_input):
    data = {"user_input": user_input}
    response = requests.post(GET_LLAMA_RESPONSE_URL, json=data)
    if response.status_code == 200:
        return response.json().get("response", "No response available.")
    else:
        st.error(f"Error: Unable to get response, {response.status_code}")
        return "Error: Unable to get response."

def get_question_recommendations(user_input=None):
    if user_input:
        data = {"user_input": user_input}
        response = requests.post(GET_QUESTION_RECOMMENDATIONS_URL, json=data)
        if response.status_code == 200:
            return response.json().get("response", {}).get("nextQuestions", [])
        else:
            st.error(f"Error: Unable to get question recommendations, {response.status_code}")
            return []
    else:
        return [
            "How Ben helps in home-buying process?",
            "How do I buy a house?",
            "What is the current real estate market trend?",
            "Can you help me with mortgage options?",
            "What should I look for in a property?"
        ]

# Typing effect for responses
def typewriter_effect(text):
    response_container = st.empty()
    displayed_text = ""
    for char in text:
        displayed_text += char
        response_container.markdown(displayed_text)
        time.sleep(0.01)

# Callback function to handle question selection
def question_callback(selected_question):
    st.session_state.selected_question = selected_question
    st.session_state.question_clicked = True
    st.session_state.initial_load = False

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
    st.session_state.initial_load = True
    st.session_state.question_clicked = False
    st.session_state.selected_question = None

# Display suggested questions
def display_suggested_questions(user_input=None):
    questions = get_question_recommendations(user_input)
    if questions:  # Only display if there are questions
        st.write("### Suggested Questions:")
        cols = st.columns(len(questions))
        for i, question in enumerate(questions):
            # Use a lambda to pass `question` to the callback
            cols[i].button(question, key=f'question_{user_input}_{i}', on_click=question_callback, args=(question,))

# Main function to handle user input, response, and recommendations
def process_user_input(user_input):
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)

    response = get_llama_response(user_input)
    if response:
        with st.chat_message("assistant"):
            typewriter_effect(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        display_suggested_questions(user_input)

# Display chat messages
def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Streamlit UI setup
st.title("Ask Ben")
display_chat_history()

# Initial load of questions
if st.session_state.initial_load:
    display_suggested_questions()

# Check if a question was clicked and process it
if st.session_state.question_clicked:
    process_user_input(st.session_state.selected_question)
    # Reset state to avoid reprocessing
    st.session_state.question_clicked = False
    st.session_state.selected_question = None

# Chat input handling
if user_input := st.chat_input("What would you like to know?"):
    process_user_input(user_input)

# Sidebar instructions
st.sidebar.header("Instructions")
st.sidebar.write("Type your message in the input box and press Enter to chat with BEN!")
