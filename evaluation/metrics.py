from sklearn.metrics import confusion_matrix

def compute_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    false_alarm_rate = fp / (fp + tn)

    return sensitivity, specificity, false_alarm_rate
