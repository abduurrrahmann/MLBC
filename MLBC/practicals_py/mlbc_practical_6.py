import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score
iris=load_iris()
X=iris.data
y=iris.target

#Graph Based Clustering

graph_cluster=SpectralClustering(n_clusters=3,affinity='nearest_neighbors',random_state=42)
cluster_labels=graph_cluster.fit_predict(X)
plt.figure(figsize=(7,5))
plt.scatter(X[:,0],X[:,1],c=cluster_labels,cmap='viridis',s=50)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Graph Based Clustering using Spectral Clustering')
plt.show()

#CART Algorithm

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
cart=DecisionTreeClassifier(criterion='gini',max_depth=3,random_state=42)
cart.fit(X_train,y_train)
y_pred=cart.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print('CART Classification Accuracy:',accuracy)
plt.figure(figsize=(12,7))
plot_tree(cart,feature_names=iris.feature_names,class_names=iris.target_names,filled=True)
plt.title('CART Decision Tree')
plt.show()