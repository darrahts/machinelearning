# machinelearning
Collection of python files and jupyter notebooks for machine learning.

## Regression notes:
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

## Classification notes:
