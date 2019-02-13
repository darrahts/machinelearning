# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


dataset = pd.read_csv("Position_Salaries.csv")

X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, (dataset.shape[1] - 1)].values

linreg = LinearRegression()
linreg.fit(X, y)

polyfeat =PolynomialFeatures(degree=4)
X_poly = polyfeat.fit_transform(X)

polyreg = LinearRegression()
polyreg.fit(X_poly, y)


#visualize true
X_grid = np.arange(min(X), max(X), .1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
#plt.plot(X, linreg.predict(X), color='yellow')
plt.plot(X_grid, polyreg.predict(polyfeat.fit_transform(X_grid)), color='blue')
plt.title("regression")
plt.show()

#predict a new result, need to see if level 6.5 equates to 160k
y_pred_lin = linreg.predict(6.5)
y_pred_poly = polyreg.predict(polyfeat.fit_transform(6.5))










