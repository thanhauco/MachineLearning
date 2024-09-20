import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from matplotlib.colors import ListedColormap

np.random.seed(7)  
low_r = 10  
high_r = 15
n = 1550
X = np.random.uniform(low=[0, 0], high=[4, 4], size=(n,2))
drop = (X[:, 0]**2 + X[:, 1]**2 > low_r) & (X[:, 0]**2 + X[:, 1]**2 < high_r)
X = X[~drop]
y = (X[:, 0]**2 + X[:, 1]**2 >= high_r).astype(int) 
colors = ['red', 'blue']
plt.figure(figsize=(6, 6))
for i in np.unique(y):
    plt.scatter(X[y==i, 0], X[y==i, 1], label = "y="+str(i), 
                color=colors[i], edgecolor="white", s=50)
circle = plt.Circle((0, 0), 3.5, color='black', fill=False,
                    linestyle="--", label="Actual boundary")
plt.xlim([-0.1, 4.2])
plt.ylim([-0.1, 5])
ax = plt.gca()  
ax.set_aspect('equal')
ax.add_patch(circle)
plt.xlabel('$x_1$', fontsize=16)
plt.ylabel('$x_2$', fontsize=16)
plt.legend(loc='best', fontsize=11)

plt.show()


#Listing 2

tree_clf = DecisionTreeClassifier(random_state=0)  
tree_clf.fit(X, y)
plt.figure(figsize=(17,12))
tree.plot_tree(tree_clf, fontsize=17, feature_names=["x1", "x2"])
plt.show()
