from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

def evaluate_model(y_test, y_pred):
    metrics = {"accuracy": accuracy_score(y_test, y_pred),
               "precision": precision_score(y_test, y_pred),
               "recall": recall_score(y_test, y_pred),
               "f1": f1_score(y_test, y_pred),
               "confusion_matrix": confusion_matrix(y_test, y_pred)}
    return metrics


def predict_with_threshold(failure_probs, threshold):
    y_pred = (failure_probs>= threshold).astype(int)
    return y_pred


def find_best_threshold(failure_probs, y_val, thresholds):
    best_threshold = None
    best_cost = float('inf')
    for threshold in thresholds:
        y_pred= predict_with_threshold(failure_probs, threshold)
        cm = confusion_matrix(y_val, y_pred,labels=[0, 1])
        fp = cm [0,1]
        fn = cm [1,0]
        total_cost =5000 * fp + 50000 * fn
        if total_cost < best_cost:
            best_threshold = threshold
            best_cost = total_cost
    return best_threshold, best_cost
