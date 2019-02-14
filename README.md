# machinelearning
Collection of python files and jupyter notebooks for machine learning. After taking a few different online machine learning courses I decided to compile everything I learned in one location.

## Regression
- Some environments will complain about the shape of y or y_pred with errors like: 

```
Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
```

or

```
ValueError: Expected 2D array, got scalar array instead:
array=6.5.
```
I didn't run into this problem until running the code on a different laptop than it was originally written on. If this happens, reshape y with `y = y.reshape(-1,1)` and reshape the value you are trying to predict (in this case its 6.5) with `[[6.5]]`

#### whats included:
- polynomial regression
- linear regression
- multivariate regression
- support vector regression
- decision tree
- random forest
- Position_Salaries.csv


## Classification

#### whats included:
- K-nearest-neighbors
- naive-bayes
- decision tree
- random forest (Jupyter Notebook)
- support vector machine
- logistic regression (as a binary classifier)
- Social_Network_Ads.csv


## Clustering:
**jupyter notebooks from here on**
- hierarchical clustering
- k-means clustering
- Mall_Customers.csv


## Association Rule Learning

#### whats included:
- apriori
- apyori (library which implements apriori)
- Market_Basket_Optimisation.csv
- store_data.csv

## Reinforcement Learning

#### whats included:
- reinforcement learning
- Ads_CTR_Optimisation.csv

## Natural Language Processing

#### Whats included:
- nlp bag of words
- Restaurant_Reviews.tsv

## Dimensionality Reduction

#### Whats included:
- dim reduction PCA LDA
- kernel PCA
- Wine.csv 

## Models and Boosting

#### Whats included:
- xgboost
- cross validation
- grid search (hyperparameters)
- random search (hyperparameters)
- Churn_Modelling.csv
- Social_Network_Ads.csv

## Fraud Detection
 - this is NOT my original code, I can't remember where I found it from (I think kaggle). 






