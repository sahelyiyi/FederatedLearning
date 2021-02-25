import numpy as np
from sklearn.metrics import mean_squared_error


def nmse_func(Y, Y_pred):
    MSE = mean_squared_error(Y, Y_pred)
    NMSE = MSE / np.square(Y).mean()
    return NMSE
