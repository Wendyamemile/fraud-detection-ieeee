from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from threshold import find_best_threshold

def evaluate_model(model, X_val, y_val):
    probs = model.predict_proba(X_val)[:, 1]

    roc_auc = roc_auc_score(y_val, probs)
    pr_auc = average_precision_score(y_val, probs)

    print("ROC-AUC:", roc_auc)
    print("PR-AUC:", pr_auc)

    # best threshold
    best_thresh = find_best_threshold(y_val, probs)

    preds = (probs > best_thresh).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_val, preds))

    return roc_auc, pr_auc

def plot_confusion_matrix(model, X_val, y_val, threshold=0.5):
    probs = model.predict_proba(X_val)[:, 1]
    preds = (probs > threshold).astype(int)

    cm = confusion_matrix(y_val, preds)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.show()

    return cm