from keras.datasets import reuters
import numpy as np
import DL_function_lib as dlib
from keras import models, layers
import matplotlib.pyplot as plt

class DL_Reuters_Model():
    def __init__(self):

        #The downloaded data supplies train and test data of reuters news articles. A single dataset consists of the 
        #article itself resp. a list of integers, where each integers represent a word index(analog to the imdb_model)
        #and a label which can an integer between 0 and 45, each representing a article category.
        #The task for the model obviously is to predict the article category.
        #In contrast to the previous example we not only have 2 possible classifications, but 46,
        # which is reflected in the choice of a different loss function and a different activation function of
        #our output layer.
        (self.train_data, self.train_labels), (self.test_data, self.test_labels) = reuters.load_data(num_words=10000)

        self.word_index = reuters.get_word_index()

        #If you remove the comment tags in the comment right below, you get a decoded(meaning retransformed into strings resp. words)
        # film review. Note that only the 10.000 most common words have been downloaded, so it will most likely not print out every word.
        #The decoded review is accessed with train_data[x] so for a different review you need to change the index.

        # reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
        # decoded_article = " ".join(reverse_word_index.get(i+1, "?") for i in train_data[0])
        # print(decoded_article)

        #This time, for future convenience, I outsourced the functions for vectorizing the sequences
        #and for one-hot coding the labels in an own module, where I will collect deep learning
        #functions I expect to use regullary in later projects.
        self.x_train = dlib.vectorize_sequences(self.train_data)
        self.x_test = dlib.vectorize_sequences(self.test_data)

        #The one hot coding in this example is done analog to the imdb model. I also use the previous function
        #but change the dimension parameter, which is by default 10.000 to 46 to fit the 46 different article categories.
        self.one_hot_train_labels = dlib.vectorize_sequences(self.train_labels, 46)
        self.one_hot_test_labels =  dlib.vectorize_sequences(self.test_labels, 46)

        self.model = models.Sequential()
        self.model.add(layers.Dense(64, activation="relu", input_shape=(10000,)))
        self.model.add(layers.Dense(64, activation="relu"))
        self.model.add(layers.Dense(46, activation="softmax"))
        self.model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

        self.x_val = self.x_train[:1000]
        self.partial_x_train = self.x_train[1000:]
        self.y_val = self.one_hot_train_labels[:1000]
        self.partial_y_train = self.one_hot_train_labels[1000:]

        if __name__=="__main__":
            self.history = self.model.fit(self.partial_x_train, self.partial_y_train, epochs=9, batch_size=512, validation_data=(self.x_val,self.y_val))

            dlib.loss_and_correctclassification_plot(self.history)
        
        else:
            self.model.fit(self.partial_x_train, self.partial_y_train, epochs=9, batch_size=512)

if __name__=="__main__":
    reuters_model = DL_Reuters_Model()
    predictions = reuters_model.model.predict(reuters_model.x_test)
    category = np.argmax(predictions[0])
    print(category)