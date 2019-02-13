# Decision Tree Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn import tree

from IPython.display import Image  
from sklearn.externals.six import StringIO  
import pydotplus


dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:, 2:4].values
y = dataset.iloc[:, (dataset.shape[1]-1)].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0)

#decision trees are not based on euclidean distance, so do not need scaling

clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
features = list(dataset.columns[[2,3]])
dot_data = StringIO()  
tree.export_graphviz(clf, out_file=dot_data,feature_names=features)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())  

# Visualising the Training set results
# =============================================================================
#X_set, y_set = X_train, y_train
#X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = #X_set[:, 0].max() + 1, step = 0.1),
#                     np.arange(start = X_set[:, 1].min() - 1, stop = #X_set[:, 1].max() + 1, step = 0.1))
#plt.contourf(X1, X2, clf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
#plt.xlim(X1.min(), X1.max())
#plt.ylim(X2.min(), X2.max())
#for i, j in enumerate(np.unique(y_set)):
#    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                c = ListedColormap(('red', 'green'))(i), label = j)
#plt.title('Decision Tree Classification (Training set)')
#plt.xlabel('Age')
#plt.ylabel('Estimated Salary')
#plt.legend()
#plt.show()
# =============================================================================

# Visualising the Test set results
#X_set, y_set = X_test, y_test
#X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
#plt.contourf(X1, X2, clf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
#plt.xlim(X1.min(), X1.max())
#plt.ylim(X2.min(), X2.max())
#for i, j in enumerate(np.unique(y_set)):
#    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                c = ListedColormap(('red', 'green'))(i), label = j)
#plt.title('Decision Tree Classification (Test set)')
#plt.xlabel('Age')
#plt.ylabel('Estimated Salary')
#plt.legend()
#plt.show()
