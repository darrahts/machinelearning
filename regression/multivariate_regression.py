# -*- coding: utf-8 -*-


'''import libraries'''
import numpy as np
np.set_printoptions(threshold=np.nan)
import matplotlib.pyplot as plt
import pandas as pd
    #for handling: missing data, categorical data, dummy encoder, scaling features
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import statsmodels.formula.api as sm

'''import dataset'''
dataset = pd.read_csv("50_Startups.csv")

'''separate features from target'''
#[:,:-1] take all rows, take all columns except the last one
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, (dataset.shape[1] -1)].values


'''encode categorical data using a dummy encoder
    ex. 3 labels, a,b,c; encoded becomes [1,0,0], [0,1,0], [0,0,1]'''
labelEncoder = LabelEncoder()
X[:,3] = labelEncoder.fit_transform(X[:,3])
oneHotEncoder = OneHotEncoder(categorical_features = [3])
X = oneHotEncoder.fit_transform(X).toarray()

'''avoiding the dummy varible trap
    i.e. removing one of the dummy variables since 1 - P = P^-1'''
X = X[:, 1:]


'''train_test_split'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred1 = regressor.predict(X_test)
plt.plot(y_test)
plt.plot(y_pred1)
plt.show()

'''backward elimination of features that do not have significance
    need to add a feature for b0 due to statsmodels api implementaiton'''
X = np.append(arr=np.ones((50,1)).astype(int), values=X, axis=1)

#optimal features of X
X_opt = X[:, [0,1,2,3,4,5]]

#select significance level (P-value)
sl = .08

#fit model with all predictors
reg_ols = sm.OLS(endog=y, exog=X_opt).fit()

#remove feature with highest P value
reg_ols.summary()
X_opt = X[:, [0,1,3,4,5]]

#repeat
reg_ols = sm.OLS(endog=y, exog=X_opt).fit()
reg_ols.summary()
X_opt = X[:, [0,3,4,5]]

#repeat
reg_ols = sm.OLS(endog=y, exog=X_opt).fit()
reg_ols.summary()
X_opt = X[:, [0,3,5]]

#repeat
reg_ols = sm.OLS(endog=y, exog=X_opt).fit()
reg_ols.summary()

X_test_opt = np.append(arr=np.ones((10,1)).astype(int), values=X_test, axis=1)
X_test_opt = X_test_opt[:, [0,3,5]]
y_pred2 = reg_ols.predict(X_test_opt)

#backward elimination with p values and adjusted r squared
def backwardElimination(x, y, sl):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, y, sl)


#backward elimination with p values only
def backwardElimination1(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, sl)


















