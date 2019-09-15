# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap



#get the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, 2:4].values
y = dataset.iloc[:, (dataset.shape[1]-1)].values

#view the dataset
#plt.scatter(X[:, 0], y, c='r', marker='.')
#plt.scatter(X[:, 1], y, c='g', marker='.')
#plt.show()


#split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0)

#scale the features
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#fit to logistic regression classifier
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

#predict the test results
y_pred = classifier.predict(X_test)

#confusion matrix
cm = confusion_matrix(y_test, y_pred)


import seaborn as sns
sns.regplot(x=X_test, y=y_pred, logistic=True)

#visualize the graph (Train set)
X1, X2 = np.meshgrid(np.arange(start=X_train[:,0].min()-1, stop=X_train[:,0].max()+1, step=.01),
                     np.arange(start=X_train[:,0].min()-1, stop=X_train[:,0].max()+1, step=.01))
y_pred_step = classifier.predict(np.array([X1.ravel(), X2.ravel()]).T)
plt.contourf(X1, X2, y_pred_step.reshape(X1.shape), alpha=.75, cmap=ListedColormap(('red', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_train)):
    plt.scatter(X_train[y_train==j, 0], X_train[y_train==j, 1],
                c=ListedColormap(('red', 'blue'))(i), label=j)

plt.title("logistic regression (training set)")
plt.xlabel("age")
plt.ylabel("salary")
plt.legend()
plt.show()


#visualize the graph (Test set)
X1, X2 = np.meshgrid(np.arange(start=X_test[:,0].min()-1, stop=X_test[:,0].max()+1, step=.01),
                     np.arange(start=X_test[:,0].min()-1, stop=X_test[:,0].max()+1, step=.01))
y_pred_step = classifier.predict(np.array([X1.ravel(), X2.ravel()]).T)
plt.contourf(X1, X2, y_pred_step.reshape(X1.shape), alpha=.75, cmap=ListedColormap(('red', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_test)):
    plt.scatter(X_test[y_test==j, 0], X_test[y_test==j, 1],
                c=ListedColormap(('red', 'blue'))(i), label=j)

plt.title("logistic regression (test set)")
plt.xlabel("age")
plt.ylabel("salary")
plt.legend()
plt.show()
    

