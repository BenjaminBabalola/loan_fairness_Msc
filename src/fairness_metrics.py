# src/fairness_metrics.py
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def _positive_rate(labels):
    return (labels == 1).mean()

def statistical_parity_difference(y_true, y_pred, sensitive_feature, privileged_value):
    """
    SPD = P(Y_hat=1 | unprivileged) - P(Y_hat=1 | privileged)
    y_pred: numpy array of predicted labels (0/1)
    sensitive_feature: array-like same length as y_pred (group membership)
    privileged_value: value in sensitive_feature considered privileged
    """
    sf = np.array(sensitive_feature)
    pred = np.array(y_pred)
    priv_mask = sf == privileged_value
    unpriv_mask = ~priv_mask
    p_priv = pred[priv_mask].mean() if priv_mask.sum()>0 else np.nan
    p_unpriv = pred[unpriv_mask].mean() if unpriv_mask.sum()>0 else np.nan
    return p_unpriv - p_priv

def disparate_impact(y_true, y_pred, sensitive_feature, privileged_value):
    sf = np.array(sensitive_feature)
    pred = np.array(y_pred)
    priv_mask = sf == privileged_value
    unpriv_mask = ~priv_mask
    p_priv = pred[priv_mask].mean() if priv_mask.sum()>0 else np.nan
    p_unpriv = pred[unpriv_mask].mean() if unpriv_mask.sum()>0 else np.nan
    return np.nan if p_priv==0 else p_unpriv / p_priv

def true_positive_rate(y_true, y_pred):
    # returns TPR
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return tp / (tp + fn) if (tp + fn) > 0 else np.nan

def equal_opportunity_difference(y_true, y_pred, sensitive_feature, privileged_value):
    sf = np.array(sensitive_feature)
    y_true = np.array(y_true)
    pred = np.array(y_pred)
    priv_mask = sf == privileged_value
    unpriv_mask = ~priv_mask
    tpr_priv = true_positive_rate(y_true[priv_mask], pred[priv_mask]) if priv_mask.sum()>0 else np.nan
    tpr_unpriv = true_positive_rate(y_true[unpriv_mask], pred[unpriv_mask]) if unpriv_mask.sum()>0 else np.nan
    return tpr_unpriv - tpr_priv

def group_approval_rates(y_true, sensitive_feature):
    df = pd.DataFrame({'y': y_true, 'sf': sensitive_feature})
    return df.groupby('sf')['y'].mean().reset_index().rename(columns={'y':'approval_rate'})
