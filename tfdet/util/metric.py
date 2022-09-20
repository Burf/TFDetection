import tensorflow as tf
import numpy as np

def get_threshold(y_true, y_pred):
    try:
        from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
    except Exception as e:
        print("If you want to use 'get_threshold', please install 'scikit-learn 0.14▲'")
        raise e
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    f1 = np.divide(2 * precision * recall, precision + recall, out = np.zeros_like(precision), where = (precision + recall) != 0)
    threshold = thresholds[np.argmax(f1)]
    return threshold
