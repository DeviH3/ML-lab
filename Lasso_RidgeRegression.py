#9Regularization
from sklearn.linear_model import Lasso
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
diabetes=load_diabetes()
features = diabetes.data
target = diabetes.target
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)
regression = Lasso(alpha =0.5)
model =regression.fit(features_standardized, target)
print(model.coef_)
from sklearn.linear_model import Ridge
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
boston=load_diabetes()
features = diabetes.data
target = diabetes.target
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)
regression = Ridge(alpha =0.5)
model =regression.fit(features_standardized, target)
print(model.coef_)
