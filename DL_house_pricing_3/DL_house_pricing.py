from keras.datasets import boston_housing
from keras import layers, models
import DL_function_lib as dlib
import numpy as np
import matplotlib.pyplot as plt

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

train_data, test_data = dlib.normalization(train_data, test_data)

#Creating a function for building the model, so that it can be reused in a loop dooring k-fold validation
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation="relu", input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation="relu"))
    #Output layer has no activation function to not limit its value range
    model.add(layers.Dense(1))
    #As loss function we use the mean squared error, which is a typical loss function for
    #regression tasks
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    return model

#The outcommented code is used for the k-fold validation.
#i loops k-times, hence its name, and creates k different models.
#In each loop the training data gets sliced into k-1 subsets for training and 1 subset for
#validation with each subset having the size for 1/k of the original data.
#From the plotted data we extract the optimal parameters with which we train our final model.

# k=4
# num_val_samples = len(train_data) // k
# num_epochs = 80
# all_mae_history = []


# for i in range(k):
#     print('Durchlauf #', i)
#     val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
#     val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

#     partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]], axis=0)

#     partial_train_targets = np.concatenate([train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)

#     model = build_model()

#     history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets), epochs=num_epochs, batch_size=16, verbose=0)
#     print(history.history)
#     mae_history = history.history["val_mae"]
#     all_mae_history.append(mae_history)

#     val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)

# average_mae_history = [np.mean([x[i] for x in all_mae_history]) for i in range(num_epochs)]

# smooth_mae_history=dlib.smooth_curve(average_mae_history)[10:]

# plt.plot(range(1,len(smooth_mae_history)+1),smooth_mae_history)
# plt.xlabel("Epochen")
# plt.ylabel("Mean absolute error validation")
# plt.show()

model = build_model()
model.fit(train_data, train_targets, epochs=80, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)

print(f"Mean squared error score: {test_mse_score}")
print(f"Mean absolute error score: {test_mae_score}")

