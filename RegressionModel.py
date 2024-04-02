#3Regression model
import pandas as pd
dataset=pd.read_csv('/home/lab-7/Downloads/Realestate.csv')
dataset
x = dataset.iloc[:,[2,3,4]]
y = dataset.iloc[:,-1]
print(x)
print(y)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.1,
random_state=0)
from sklearn.linear_model import LinearRegression
regr = LinearRegression()
regr.fit(x_train, y_train)
print(regr.score(x_test, y_test))
y_pred = regr.predict(x_test)
print(y_pred)
