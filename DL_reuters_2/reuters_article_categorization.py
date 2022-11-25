import DL_reuters as dlr
import numpy as np

article_categorizer = dlr.DL_Reuters_Model()
predictions = article_categorizer.model.predict(article_categorizer.x_test)
category = np.argmax(predictions[0])
print(category)