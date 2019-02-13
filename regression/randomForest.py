# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


dataset = pd.read_csv("Position_salaries.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


regressor = RandomForestRegressor(n_estimators=200, random_state=0)

regressor.fit(X, y)

y_pred = regressor.predict(6.5)


x_steps = np.arange(min(X), max(X), .01)
x_steps = x_steps.reshape((len(x_steps), 1))
plt.scatter(X, y, color='red')
plt.plot(x_steps, regressor.predict(x_steps), color='blue')
plt.title("salary predictions")
plt.xlabel("position")
plt.ylabel("salary")
plt.show()