# -*- coding: utf-8 -*-

#information entropy

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, (dataset.shape[1] - 1)].values

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

y_pred = regressor.predict(6.5001)


x_steps = np.arange(min(X), max(X), .01)
x_steps = x_steps.reshape((len(x_steps), 1))
plt.scatter(X, y, color='red')
plt.plot(x_steps, regressor.predict(x_steps), color='blue')
plt.title("salary predictions")
plt.xlabel("position")
plt.ylabel("salary")
plt.show()