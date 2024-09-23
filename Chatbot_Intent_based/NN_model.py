import json
import pickle
import tensorflow as tf
from tensorflow.python.keras.models import load_model

# Load the model and intents with error handling
def load_model_and_intents(): 
    try:
        model = load_model('NN_model.h5')
        print("Model loaded")
    except Exception as e:
        print("Error loading model:", e)
        exit()

    try:
        with open('Intents/intents.json', 'r', encoding='utf-8') as f:
            intents = json.load(f)
        with open('Intents/words.pkl', 'rb') as f:
            words = pickle.load(f)
        with open('Intents/classes.pkl', 'rb') as f:
            classes = pickle.load(f)
    except Exception as e:
        print("Error loading files:", e)
        exit()

    return model, intents, words, classes