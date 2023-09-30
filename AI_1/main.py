import pandas as pd
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

data = pd.read_csv('./res/computer.dat', sep='\s+')

data.isnull().any()

data = data.drop(['FB', 'SEX', 'ALTER', 'HERKU', 'V5A', 'V5B'], axis=1)

data.describe()

X = data.iloc[:].values

dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.ylabel('Euclidean distances')
plt.xlabel('Clusters')
plt.show()

hc = AgglomerativeClustering(n_clusters = 3, metric = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

plt.title('Multidimensional scaling')
plt.scatter(X[:, 0], X[:, 1], c = hc.labels_, cmap='rainbow')
plt.show()

wcss = []
for i in range(2, 6):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=3, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(2, 6), wcss)
plt.title('The Elbow Method')
plt.xlabel('Clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters = 3, init = 'k-means++', n_init=3 ,random_state = 42)
y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c = kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Multidimensional scaling')
plt.show()