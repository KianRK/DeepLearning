In terms of the data structure, this project is very similar to the previous one in which we classified movie reviews.
The training data again is a list of integers, each representing a word index, so that each integer has unique word assigned to it and vice versa ( a bijective function so to say or a non redundant dictionary in python terms).
The training data however, instead of movie reviews, now represents reuters news articles, each belonging to a single news category like sports, politics, science etc.
So now the labels do not anymore consist merely of binary values of 0 and 1, but are integers between 0 and 45, representing 46 different news categories, each article is uniquely assigned to.

The task now is to create a model that learns how to classify a given article to the correct category.

This kind of problem is known as categorial classification, and differentiates from the binary classification in choice of a different loss function (categorial_crossentropy)
and a different activation function (softmax) for the output layer.Since the softmax function assigns probabilities to each class, which is exactly what we want to ascertain for our prediction, the output vector naturally has 46 dimensions, instead of 1 when we use the sigmoid function.

As a little twist I started to outsource reoccuring functions in a library DL_function_lib.py which I hope will turn out useful in future endeavours.
