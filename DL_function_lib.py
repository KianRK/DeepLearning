import numpy as np
import matplotlib.pyplot as plt

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1
    return results

#If we train a model with different features of varying shapes, it is advantageous to
#normalize those features values to avoid overfitting to certain features.
#This is simply achieved by substracting the mean value of each element and dividing by
#the standard deviation, so that values center around zero and have a standard deviation of 1.
def normalization(train_vect, test_vect):
    mean = train_vect.mean(axis=0)
    train_vect -= mean
    std_deviation = train_vect.std(axis=0)
    train_vect /= std_deviation
    test_vect -= mean
    test_vect /= std_deviation
    return train_vect, test_vect


#This method can be used to plot loss and and correct classification rate
# during training and validation.
#Parameters are the model history as well as training and validation metrics, which
#are accuracy by default. Later I will add automatic axis labeling depending of the
#passed metrics.
def loss_and_correctclassification_plot(hist,metric="accuracy", val_metric="val_accuracy"):
    loss = hist.history["loss"]
    val_loss = hist.history["val_loss"]
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, "bo", label = "Verlust Training")
    plt.plot(epochs, val_loss, "b", label = "Verlust Validierung")
    plt.title("Wert der Verlustfunktion Training/Validierung")
    plt.xlabel("Epochen")
    plt.ylabel("Wert der Verlustfunktion")
    plt.legend()
    plt.show()

    plt.clf()
    acc = hist.history[metric]
    val_acc = hist.history[val_metric]
    plt.plot(epochs, acc, "bo", label = "Training")
    plt.plot(epochs, val_acc, "b", label = "Validierung")
    plt.title("Korrektklassifizierungsrate Training/Validierung")
    plt.xlabel("Epochen")
    plt.ylabel("Korrektklassifizierungsrate")
    plt.legend()
    plt.show()

#This method is used to flatten a a curve by reducing the difference between 
#the point n and point n+1 to 1-factor of its original difference.
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous*factor + point*(1-factor))
        else:
            smoothed_points.append(point)
    return smoothed_points
