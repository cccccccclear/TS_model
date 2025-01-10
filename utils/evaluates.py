import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def R_squared(pred, true):
    mean_pred = np.mean(pred)
    mean_true = np.mean(true)
    tss = np.sum((true - mean_true) ** 2)
    rss = np.sum((pred - true) ** 2)
    r_squared = 1 - (rss / tss)
    
    return r_squared

def ACCU(pred, true):
    return np.mean(pred*true > 0)



def evaluate(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    r_squared = R_squared(pred, true)
    accu = ACCU(pred, true)

    return mae, mse, rmse, mape, mspe, r_squared, accu
