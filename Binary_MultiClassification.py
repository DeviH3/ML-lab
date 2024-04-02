#Binary Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
iris =datasets.load_iris()
features = iris.data[:100,:]
target =iris.target[:100]
scaler =StandardScaler()
features_standardized =scaler.fit_transform(features)
logistic_regression = LogisticRegression(random_state=0)
model = logistic_regression.fit(features_standardized,target)
y_pred=model.predict(features_standardized)
print(metrics.accuracy_score(y_pred,target))

#Multinomial Logistic Regression
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets, metrics
from sklearn.linear_model import LogisticRegression
iris=datasets.load_iris()
iris_data=iris.data
iris_data=pd.DataFrame(iris_data,columns=iris.feature_names)
iris_data['species']=iris.target
iris_data['species'].unique()
features =iris.feature_names
target ='species'
X=iris_data[features]
y=iris_data[target]
lr_iris=LogisticRegression() # default value for multi class problem is multinomial
lr_iris =lr_iris.fit(X,y)
y_pred=lr_iris.predict(X)
print(metrics.accuracy_score(y_pred,y))

#KNN
data=pd.read_csv("/home/lab-7/Downloads/Social_Network_Ads.csv") 
print(data.head()) 
X = data.iloc[:, [2, 3]].values 
y = data.iloc[:, 4].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,random_state = 0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, y_train) 

y_pred = classifier.predict(X_test)
print(y_pred) 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
