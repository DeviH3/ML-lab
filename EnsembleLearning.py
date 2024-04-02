import numpy as np
import pandas as pd
data=pd.read_csv("Wine.csv")
cols=['Alcohol','Color_Intensity','Proline','Ash_Alcanity']
from sklearn.preprocessing import MinMaxScaler
mms=MinMaxScaler(feature_range=(0,1))
data[cols]=mms.fit_transform(data[cols])
x=data.drop('Customer_Segment',axis=1)
y=data['Customer_Segment']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
preds=clf.predict(X_test)
preds
from sklearn.metrics import accuracy_score
accuracy_test_DT=accuracy_score(y_test,preds)
train_preds=clf.predict(X_train)
accuracy_train_DT=accuracy_score(y_train,train_preds)
print('accuracy_train_DT :',accuracy_train_DT)
print('accuracy_test_DT :',accuracy_test_DT) 
