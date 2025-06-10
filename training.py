import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)


for intent in intents['intents']:
    for pattern in intent['patterns']:


        w = nltk.word_tokenize(pattern)
        words.extend(w)

        documents.append((w, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))


pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))


training = []
output_empty = [0] * len(classes)
i=0
for doc in documents:

    bag = []

    pattern_words = doc[0]

    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
    


random.shuffle(training)


inputs = [np.array(input_data) for input_data, _ in training]
outputs = [np.array(output_data) for _, output_data in training]

inputs_array = np.array(inputs)
outputs_array = np.array(outputs)


train_x = list(inputs_array)
train_y = list(outputs_array)
print("Training data created")


model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(len(train_y[0]), activation='softmax'))


sgd = SGD(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


chatbot_model = model.fit(np.array(train_x), np.array(train_y), epochs=300, batch_size=60)
model.save('chatbot_model.h5', chatbot_model)

print("model created")