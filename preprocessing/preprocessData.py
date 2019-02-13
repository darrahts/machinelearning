# -*- coding: utf-8 -*-


'''import libraries'''
import numpy as np
np.set_printoptions(threshold=np.nan)
import matplotlib.pyplot as plt
import pandas as pd
    #for handling missing data, categorical data, dummy encoder, scaling features
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


'''import dataset'''
dataset = pd.read_csv("Data.csv")


'''separate features from target'''
#[:,:-1] take all rows, take all columns except the last one
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, (dataset.shape[1] -1)].values


'''handle missing data by taking mean of column
    or mean of column where other features are the same (more complex)'''
imputer = Imputer()
#imputer = imputer.fit(X[:,1:3])
#X[:,1:3] = imputer.transform(X[:, 1:3])
#X[:,1:3] = imputer.fit(X[:,1:3]).transform(X[:,1:3])
X[:,1:3] = imputer.fit_transform(X[:,1:3])


'''encode categorical data using a dummy encoder
    ex. 3 labels, a,b,c; encoded becomes [1,0,0], [0,1,0], [0,0,1]'''
labelEncoder = LabelEncoder()
X[:,0] = labelEncoder.fit_transform(X[:,0])
oneHotEncoder = OneHotEncoder(categorical_features = [0])
X = oneHotEncoder.fit_transform(X).toarray()
y = labelEncoder.fit_transform(y)


'''feature scaling to move the range of multiple features closer together 
    to affect the euclidean distance (or other distance metric) of the features
    standardization: xs = (x - mean) / sd(x)
    normalization:   xn = (x - min) / (max - min)'''
scaler = StandardScaler()
X = scaler.fit_transform(X)


'''train_test_split'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 0)









#just for viewing in the variable explorer
#dfX = pd.DataFrame(X)
#dfy = pd.DataFrame(y)








































