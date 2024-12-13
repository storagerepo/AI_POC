import streamlit as st
from agent import Chatbot


# Initialize the chatbot
chatbot = Chatbot()

# Streamlit UI setup
st.title("Real Estate AI Assistant")
st.write("Chat with the assistant to get real estate information!")

# Display chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Input text box
user_input = st.text_input("You: ", "")

# Handle user input
if user_input:
    # Show user input in the chat history
    st.session_state.history.append(f"You: {user_input}")
    
    # Get chatbot's response
    try:
        response = chatbot.run(user_input)
        st.session_state.history.append(f"Assistant: {response}")
    except Exception as e:
        st.session_state.history.append(f"Assistant: Sorry, something went wrong. {e}")

# Display the chat history
for message in st.session_state.history:
    st.write(message)


