import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test set.

    Args:
        model: Trained Keras model.
        X_test: Test images.
        y_test: True labels for the test images.

    Returns:
        Dictionary: Contains evaluation metrics (accuracy, precision, recall, f1 score, mAP).
    """
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Calculate Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    mAP = calculate_map(y_test, y_pred_probs)

    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, mAP: {mAP:.4f}")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "mAP": mAP}

def calculate_map(y_true, y_pred_probs):
    """Calculate the mean Average Precision (mAP)."""
    n_classes = y_true.shape[1]
    APs = [average_precision_score(y_true[:, i], y_pred_probs[:, i]) for i in range(n_classes)]
    return np.mean(APs)
