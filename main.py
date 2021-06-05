import nltk
import numpy as np
import json
import random
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, Flatten
import keras

lemmatizer = WordNetLemmatizer()

# we are just loading the data here
def load_data(filename):
    try:
        data = json.load(open(filename, ))
    except:
        data = None
    return data


def data_processing(data,questionKey,tagKey):
    words = []
    classes = []
    documents = []
    ignore_letters = ["?", "!", ",", "."]
    for intent in data:
        for pattern in intent[questionKey]:
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append([word_list, intent[tagKey]])
        if intent[tagKey] not in classes:
            classes.append(intent[tagKey])
    words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
    words = sorted(set(words))
    classes = sorted(set(classes))
    training = []
    output_empty = [0] * len(classes)

    for document in documents:
        bag = []

        word_patterns = document[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
        print(word_patterns)

        for word in words:

            if word in word_patterns:
                bag.append(1)
            else:
                bag.append(0)

        output_row = list(output_empty)
        output_row[classes.index(document[1])] = 1

        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training)
    train_x = list(training[:, 0])
    train_y = list(training[:, 1])
    return np.array(train_x), np.array(train_y), words, document, classes


def train_model(hidden_layers: [[128, "relu"], [234, "relu"]],trainX,trainY):
    model = Sequential()
    model.add(Flatten())
    model.add(
        Dense(
            128,
            input_shape=(len(trainX[0]),),
            activation="relu"
        )
    )
    for i in hidden_layers:
        model.add(Dense(i[0],i[1]))

    model.add(Dense(len(trainY[0]), activation="softmax"))
    model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
    return model


def predict_class(model,sentence, labels, words):
    sentence_word = nltk.word_tokenize(sentence)
    sw = [lemmatizer.lemmatize(word) for word in sentence_word]
    bag = [0] * len(words)
    for w in sw:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    bow =  np.array(bag)
    res = model.predict(np.array([bow]))[0]
    results = [[i,r] for i, r in enumerate(res) if r>0.25]
    results.sort(key = lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent":labels[r[0]], "probability":str(r[1])})
    return return_list


