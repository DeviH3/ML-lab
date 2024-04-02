d = pd.read_csv('/home/lab-7/Downloads/wineQualityReds.csv')
d.head()
x = d.iloc[:,:-1]
y = d.iloc[:,-1]
from sklearn.preprocessing import MinMaxScaler
s = MinMaxScaler()
x = pd.DataFrame(s.fit_transform(x),columns = x.columns)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 8)
X_train = pca.fit_transform(x_train)
X_test = pca.transform(x_test)
explained_variance = pca.explained_variance_ratio_
print(explained_variance)
from sklearn.neighbors import KNeighborsClassifier
KNN_mod = KNeighborsClassifier(n_neighbors=10)
KNN_mod.fit(X_train,y_train)
pred = KNN_mod.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
print(accuracy_score(y_test,pred)*100) 

#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=5)
X_train = lda.fit_transform(x_train,y_train)
X_test = lda.transform(x_test)
explained_variance = lda.explained_variance_ratio_
print(explained_variance)
from sklearn.neighbors import KNeighborsClassifier
KNN_mod = KNeighborsClassifier(n_neighbors=10)
KNN_mod.fit(X_train,y_train)
pred = KNN_mod.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
print(accuracy_score(y_test,pred)*100)
