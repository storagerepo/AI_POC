import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import streamlit as st
import time

from NN_model import load_model_and_intents
from faiss_handler import initialize_faiss_indexs, retrieve_past_context, store_conversation
from llama_api import llama_api


model, intents, words, classes = load_model_and_intents()

#Tokenizes and lemmatizes the input sentence
lemmatizer = WordNetLemmatizer()
def clean_up_sentence(sentence):
  
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

#Create bag of words aray for NN Model
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return np.array(bag)


#Predicting class from NN Model
def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.98
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


#Getting response from NN or Llama API
def getResponse(ints, intents_json, full_message):
    if ints:  
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                return result
    
    # If no matching INTENT from NN model, call Llama API for response.
    return llama_api(full_message)  


# Modify chatbot response function to include context
def chatbot_response(msg):
    past_indices = retrieve_past_context(msg)
    
    if past_indices[0][0] == -1:  
        context = ""  
    else:
        valid_indices = [i for i in past_indices[0] if i >= 0 and i < len(st.session_state.conversation_history)]
        context = " ".join([st.session_state.conversation_history[i]['user_message'] for i in valid_indices])
        
    full_message = f"Previous conversation about this from Vector DB: {context}. Current query: {msg}"
    ints = predict_class(msg, model)
    res = getResponse(ints, intents, full_message)
    store_conversation(msg, res)
    return res



# Chatbot GUI Setup using Streamlit
st.title("Ask Ben")

def type_response(response):
    bot_message = ""
    message_placeholder = st.empty()  
    for char in response:
        bot_message += char
        message_placeholder.markdown(f"Ben: {bot_message}")  
        time.sleep(0.01)  

initialize_faiss_indexs()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    response = chatbot_response(prompt)  
    
    with st.chat_message("assistant"):
        type_response(response)  

    st.session_state.messages.append({"role": "assistant", "content": response})


st.sidebar.header("Instructions")
st.sidebar.write("Type your message in the input box and press Enter to chat with BEN!")
