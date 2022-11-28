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

def normalization(train_vect, test_vect):
    mean = train_vect.mean(axis=0)
    train_vect -= mean
    std_deviation = train_vect.std(axis=0)
    train_vect /= std_deviation
    test_vect -= mean
    test_vect /= std_deviation
    return train_vect, test_vect


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