#validation curve
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve
df = pd.read_csv("F:\ML\ml datasets\wineQualityReds.csv") 
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

rf = RandomForestClassifier(n_estimators=100, criterion='gini')
train_score, test_score=validation_curve(rf, X, y,param_name="max_depth",param_range=np.arange(1, 11), cv=10,scoring="accuracy")
print(validation_curve(rf, X, y,param_name="max_depth",param_range=np.arange(1, 11), cv=10,scoring="accuracy"))

#Plotting  
import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd
param_range=np.arange(1, 11)
mean_train_score = np.mean(train_score, axis = 1)
print(mean_train_score)
mean_test_score = np.mean(test_score, axis = 1)
mtp.plot(param_range, mean_train_score, label = "Training Score", color = 'b')
mtp.plot(param_range, mean_test_score,label = "Cross Validation Score", color
= 'g')
mtp.title("Validation Curve with randomforest Classifier")
mtp.xlabel("max depth")
mtp.ylabel("Accuracy")
mtp.tight_layout()
mtp.legend(loc = 'best')
mtp.show()

