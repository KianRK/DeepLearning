from keras.datasets import imdb
import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
import matplotlib.pyplot as plt

class DL_Model():
    
    def __init__(self):
        #Loading the data and assigning it to data and label lists for training and testing. the parameter num_words specify, that only the n 
        # most frequent words are loaded.

        (self.train_data, self.train_labels), (self.test_data, self.test_labels) = imdb.load_data(num_words
        =10000)

        #Applying vectorize_sequences() to create vectorized test and training data in x_train and x_test.

        self.x_train = self.vectorize_sequences(self.train_data)
        self.x_test = self.vectorize_sequences(self.test_data)

        #train_labels and test_labels are a list with the length len(sequences) where each element
        #corresponds to a review and can either be 1 for a positive or 0 for a negative review.
        #Therefore reshaping is not necessary but the listelements are converted to type float.

        self.y_train = np.asarray(self.train_labels).astype('float32')
        self.y_test = np.asarray(self.test_labels).astype('float32')

        #Since we later use layers with one input vector and one output vector, the Sequential
        #model is appropiate.
        self.model = models.Sequential()

        #We add 3 layers to our model. 2 With 16 hidden units and the relu activation function
        #and one output layer which gives out a scalar value. Therefore its activation function is
        #the sigmoid function which outputs either a 0 or a 1.
        self.model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
        self.model.add(layers.Dense(16, activation='relu'))
        self.model.add(layers.Dense(1, activation='sigmoid'))

        #We compile the configurated model with the RMSprop algorithm as an optimizer
        #and a learning rate lr of 0.001
        #As a loss function we choose binary crossentropy as a natural choice for our problem
        #of a binary classification and further choose accuracy as out evaluation metric. 
        self.model.compile(optimizer=optimizers.RMSprop(lr=0.001),
        loss=losses.binary_crossentropy,
        metrics=[metrics.binary_accuracy])

        #We further separate our training data in actual training data and validation
        #data to monitor our correct classification rate during the training process.
        self.x_val = self.x_train[:10000]
        self.partial_x_train = self.x_train[10000:]
        self.y_val = self.y_train[:10000]
        self.partial_y_train = self.y_train[10000:]

        #This part is the actual training part. We pass the training data partial_x_train and the
        #corresponding labels in partial_y_train. The epochs argument defines how often 
        #the model should filter the data through its layers.
        #In each run it trains itself with a batch size of 512 samples.
        #x_val and y_val are passed as validation data as explained above.
        #After evalutating our model we actually see that a value of 4 for epochs leads
        #to the best correct classification rate with our given model configuration.
        if __name__ == "__main__":
            
            #in history.history holds a python dictionary with data of our training process.
            #it contains data for each epoch of the training process.
            self.history = self.model.fit(self.partial_x_train, self.partial_y_train, epochs = 4,
            batch_size = 512, validation_data=(self.x_val, self.y_val))
            self.history_dict = self.history.history
            self.loss_values = self.history_dict['loss']
            self.val_loss_values = self.history_dict['val_loss']
        else:
            self.model.fit(self.x_train, self.y_train, epochs = 4, batch_size=512)

        
   

    #The following code only plots the data from history.history
    #to give us the possibility to visually evaluate the success of our training. 
    #I put it into an if-name block so that it only gets executed when this module is the executing
    #instance. 
    
        if __name__ == "__main__":
            epochs = range(1, len(self.loss_values) + 1)

            plt.plot(epochs, self.loss_values, 'bo', label = 'Verlust Training')
            plt.plot(epochs, self.val_loss_values, 'b', label = 'Verlust Validierung')
            plt.title('Wert der Verlustfunktion Training/Validierung')
            plt.xlabel('Epochen')
            plt.ylabel('Wert der Verlustfunktion')
            plt.legend()
            plt.show()

            plt.clf()
            self.acc = self.history_dict['binary_accuracy']
            self.val_acc = self.history_dict['val_binary_accuracy']

            plt.plot(epochs, self.acc, 'bo', label='Training')
            plt.plot(epochs, self.val_acc, 'b', label='Validierung')
            plt.title('Korrektklassifizierungsrate Training/Validierung')
            plt.xlabel('Epochen')
            plt.ylabel('Korrektklassifizierungsrate')
            plt.legend()
            plt.show()

        #This function (as its name suggests) is vectorizing the data to a required unified shape.
        #One must know that the movie reviews are provided as a list of integers. Each 
        #integer represents an index assigned to a certain word (for example 1 = hello, 37 = actor etc.).
        #So in our example a review is a list, variable in length, containing integers from 0 to 9999 (since in the previous step, we chose to only
        #load the 10.000 most common words). So when we later pass train_data or test_data to the function, we hand over a list of reviews.
        #In line 25 we create a len(sequences)*10.000 matrix filled with zeros and assign it to the variable results. We now have a matrix that for each review has 
        #10.000 elements potentially representing the index of the corresponding word as explained above.
        #Now we loop over the list of reviews with the built-in enumerate function, where i represents the index of a certain review and sequence a review,
        #so that results[i, sequence] = 1 assigns a 1 to each index for the corresponding occuring word in the i-th review.
        #With results we now return a matrix that displays if a certain word is used in a certain review.
        #This algorithm is generally know as one-hot encoding.

    def vectorize_sequences(self, sequences, dimension=10000):
            results = np.zeros((len(sequences), dimension))
            for i, sequence in enumerate(sequences):
                results[i, sequence] = 1.
            return results

