import DL_1_imdb_oo as dl 

#Here I created an instance of DL_Model.
#Differing from the tutorial, I made outsourced the training into a different file.
#I think it is a cleaner way to use it this way. I can not judge yet if this is
#best practice resp. how models are made quickly accessible after the training took place.
review_classifier = dl.DL_Model()

#Here we use model.predict() to let our model make predictions about a small
#sample of movie reviews. It will print out an array of floats representing the model's
#certainty in its prediction.
print(review_classifier.model.predict(review_classifier.x_test[0:11]))
