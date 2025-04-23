import numpy as np
from sklearn.metrics import mean_squared_error

def MCRMSE(y_trues, y_preds):
    """计算MCRMSE指标"""
    scores = []
    idxes = y_trues.shape[1]
    for i in range(idxes):
        y_true = y_trues[:,i]
        y_pred = y_preds[:,i]
        score = mean_squared_error(y_true, y_pred, squared=False)  # RMSE
        scores.append(score)
    mcrmse_score = np.mean(scores)
    return mcrmse_score, scores

def get_score(y_trues, y_preds):
    """便捷函数获取得分"""
    mcrmse_score, scores = MCRMSE(y_trues, y_preds)
    return mcrmse_score, scores 