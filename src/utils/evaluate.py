import torch
import numpy as np
from sklearn.metrics import classification
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score


def get_classification_report(y_true, y_pred, labels, target_names,digits = 4):
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    y_true = np.squeeze(y_true)
    pred_labels = torch.argmax(y_pred, dim = 1)
    pred_labels = pred_labels.cpu().numpy()
    report = classification_report(y_true,pred_labels, labels = labels, target_names = target_names, digits = digits)
    return report

