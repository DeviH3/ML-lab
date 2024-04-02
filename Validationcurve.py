#validation curve
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

