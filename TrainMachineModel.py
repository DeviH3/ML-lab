#13train machine learning model
from sklearn import datasets
x,y = datasets.load_iris(return_X_y=True)
from sklearn.preprocessing import MinMaxScaler
s = MinMaxScaler() 
x = pd.DataFrame(s.fit_transform(x))
x.head() 

from sklearn.neighbors import KNeighborsClassifier as KNN
model = KNN(n_neighbors=10)
model.fit(x,y)

import pickle
pickle.dump(model, open("model.pkl", 'wb'))
pickled_model = pickle.load(open('model.pkl', 'rb'))
pickled_model.predict(x.tail()) 
