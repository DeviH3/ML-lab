#11feed forward network
df=pd.read_csv("/home/lab-7/Downloads/seeds.csv",index_col=None)
print(df.shape)
df.dtypes
print(df.head())
print()
print()
print(df.isnull().sum())

X = df.iloc[:, 0:7].values
y = df.iloc[:, 7].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler
feature_scaler = StandardScaler()
X_train = feature_scaler.fit_transform(X_train)
X_test = feature_scaler.transform(X_test)
from sklearn.preprocessing import LabelBinarizer
lb=LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test=lb.fit_transform(y_test)
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(100,50), max_iter=300,activation ='relu',solver='adam',random_state=1)
clf.fit(X_train,y_train)

ypred=clf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,ypred) 
