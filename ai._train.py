# train_model.py
import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers.legacy import SGD # Use legacy for stability
import pickle

# --- 1. Initialization and Data Loading ---
lemmatizer = WordNetLemmatizer()
words = []
classes = [] # Intent tags (e.g., 'greeting', 'cloud_computing')
documents = [] # List of (word list, class tag) pairs
ignore_words = ['?', '!', '.', ',']

# Load your knowledge base
try:
    # Ensure intents.json is in the same directory as this script
    with open('intents.json') as file:
        data = json.load(file)
except FileNotFoundError:
    print("Error: intents.json not found. Cannot train model.")
    exit()

# --- 2. Data Pre-processing (Tokenization and Lemmatization) ---

for intent in data['intents']:
    for pattern in intent['patterns']:
        # Tokenize: Split the sentence into individual words
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Create a document pair
        documents.append((word_list, intent['tag']))
    
    if intent['tag'] not in classes:
        classes.append(intent['tag'])

# Lemmatize: Reduce words to their base form (e.g., 'running' -> 'run')
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
# Sort and remove duplicates to create a vocabulary
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

print(f"{len(documents)} documents (patterns)")
print(f"{len(classes)} classes (intent tags)")
print(f"{len(words)} unique words (vocabulary)")

# --- 3. Training Data Creation (Bag of Words) ---
training = []
output_empty = [0] * len(classes) # The target output vector

for doc in documents:
    bag = [] # The 'bag of words' for the current pattern
    pattern_words = doc[0]
    
    # Lemmatize the pattern words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    
    # Create the bag of words: 1 if word is present, 0 otherwise
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # Create the output row (Hot-encoding): 1 for the correct intent, 0 elsewhere
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])

# Shuffle and convert to numpy array for Keras
training = np.array(training, dtype=object) 

# Create training lists (X: features, Y: labels)
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# --- 4. Build and Compile the Neural Network Model ---
model = Sequential()
# Input layer (size is the number of unique words)
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
# Hidden layer
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
# Output layer (size is the number of intent classes)
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model (Stochastic Gradient Descent is a good optimizer for classification)
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# --- 5. Train the Model ---
print("\n--- Training Model ---")
# Epochs: number of times the model sees the data
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# --- 6. Save the Model and Data Files ---
# Save the model structure and weights
model.save('chatbot_model.h5', hist)

# Save the vocabulary (words) and classes for use in app.py
with open('words.pkl', 'wb') as f:
    pickle.dump(words, f)
    
with open('classes.pkl', 'wb') as f:
    pickle.dump(classes, f)

print("\n--- Training Complete! Model saved as 'chatbot_model.h5' ---")
