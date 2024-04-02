data=pd.read_csv("Iris.csv")
data.head()
x=data.iloc[:,3:5].values 
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,10):
    kmeans=KMeans(n_clusters = i, init = 'k-means++', max_iter = 100, n_init =10, random_state = 0).fit(x)
    wcss.append(kmeans.inertia_)
import matplotlib.pyplot as plt
plt.plot(range(1, 10), wcss, 'bx-', color='red')
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=3,init='k-means++',random_state=42)
y_predict=kmeans.fit_predict(x)
y_predict
kmeans.cluster_centers_
plt.scatter(x[y_predict == 0, 0], x[y_predict == 0, 1], s = 100, c = 'red', label ='Iris-setosa')
plt.scatter(x[y_predict == 1, 0], x[y_predict == 1, 1], s = 100, c = 'blue', label ='Iris-versicolour')
plt.scatter(x[y_predict == 2, 0], x[y_predict== 2, 1], s = 100, c = 'green', label ='Iris-virginica')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100,
c = 'black', label = 'Centroids')
plt.legend()


import numpy as np
from sklearn.metrics import silhouette_score
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
features,_=make_blobs(n_samples=1000,n_features=10,centers=2,cluster_std=
0.5,shuffle=True,random_state=1)
model=KMeans(n_clusters=2,random_state=1).fit(features)
target_predicted=model.labels_
silhouette_score(features,target_predicted) 
