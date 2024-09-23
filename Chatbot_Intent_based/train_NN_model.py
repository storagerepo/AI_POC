import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.optimizer_v1 import SGD
import random
from tensorflow.python.keras.regularizers import l2

#To dismiss warning messages
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from keras import __version__
import tensorflow.python.keras as tf_keras
tf_keras.__version__ = __version__


words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('Intents/intents.json','r',encoding='UTF-8').read()
intents = json.loads(data_file)

#Spliting words and classes
for intent in intents['intents']:
    for pattern in intent['patterns']:

        w = nltk.word_tokenize(pattern)
        words.extend(w)
    
        documents.append((w, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and lower each word and remove duplicates
lemmatizer = WordNetLemmatizer()
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# Sort classes
classes = sorted(list(set(classes)))

# Save words and classes
pickle.dump(words, open('Intents/words.pkl', 'wb'))
pickle.dump(classes, open('Intents/classes.pkl', 'wb'))

# Create our training data
training = []
# Create an empty array for our output
output_empty = [0] * len(classes)
# Training set, bag of words for each sentence
for doc in documents:
    # Initialize our bag of words
    bag = []
    # List of tokenized words for the pattern
    pattern_words = doc[0]
    # Lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # Create our bag of words array with 1 if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # Output is a '0' for each tag and '1' for the current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    # Check consistency
    if len(bag) != len(words) or len(output_row) != len(classes):
        print(f"Inconsistent lengths: bag ({len(bag)}) or output_row ({len(output_row)})")
        continue
    
    training.append([bag, output_row])

# Shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training, dtype=object)  # Use dtype=object for consistency

# Create train and test lists. X - patterns, Y - intents
train_x = np.array([item[0] for item in training])
train_y = np.array([item[1] for item in training])

print("Training data created")

# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons equal to number of intents to predict output intent with softmax
model = Sequential([
    Dense(128, input_shape=(len(train_x[0]),), activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.6),
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.6),
    Dense(len(train_y[0]), activation='softmax')
])

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fit the model
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save the model
model.save('NN_model.h5')
print("Model saved")