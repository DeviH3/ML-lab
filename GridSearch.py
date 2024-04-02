import numpy as np
import pandas as pd
pip install scikit-learn
data=pd.read_csv("Wine.csv")
x=data.drop('Customer_Segment',axis=1)
y=data['Customer_Segment']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
from sklearn.model_selection import GridSearchCV
grid_param={ 'n_estimators':[100,500,800], 'criterion':['gini','entropy'],'bootstrap':[True,False] }
gd_sr=GridSearchCV(estimator=clf,param_grid=grid_param,scoring='accuracy',cv=5)
gd_sr.fit(X_train,y_train)
best_parameters=gd_sr.best_score_
print(best_parameters)
best_result=gd_sr.best_score_
print(best_result)
