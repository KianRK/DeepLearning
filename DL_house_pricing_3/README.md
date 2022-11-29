Scalar regression and k-fold cross validation

In this example we want to predict house prices. The data consists of 506 datapoints (402 for training and 104 for testing) with each datapoint having 13 different values. Each value represents a different feature like crime rate per capita in the houses neighbourhood, number of rooms, infrastructural accesibility etc. The corresponding target data consist of a single float, representing the house price in units of 1.000 $.
To take into account the different value ranges of the single features, we also normalize the training data. This method is outsourced to my own library which I added to the dist-packages directory of my python interpreter for more convenient import handling.

In contrast to the previous example, where our model had to assign each data point to a existing discrete category, we now have no such thing as predefined 
categories and have to find a scalar value thus making it a problem of scalar regression.

Another significant difference is that we have very few datapoints compared to the previous examples.

The applied solution uses k-fold cross validation to train different models on different subsets of the provided data and averaging its metrics to extract the optimal parameters for our final model.
