#2 data pre-processing
import numpy as np
import pandas as pd
data=pd.read_csv('/home/lab-7/Downloads/dataset.csv')
print(data)
print(data.info())
print(data.isnull().sum())
newdata=data.dropna()
print(newdata)
mean_d=data['Age'].mean()
data['Age'].fillna(value=mean_d,inplace=True)
mean_d=data['Salary'].mean()
data['Salary'].fillna(value=mean_d,inplace=True)
print(data)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['Country']=le.fit_transform(data[['Age']])
data['Purchased']=le.fit_transform(data[['Purchased']])
from sklearn.preprocessing import MinMaxScaler
s=MinMaxScaler(feature_range=(0,1))
name=data.columns
data2=s.fit_transform(data)
data2=pd.DataFrame(data2,columns=name)
print(data2)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
bestfit=SelectKBest(score_func=chi2,k=3)
fit=bestfit.fit(data2.iloc[:,0:-1],data2.iloc[:,-1])
pd.DataFrame({"columns":["Country","Age","Salary"], "Scores" :fit.scores_})
