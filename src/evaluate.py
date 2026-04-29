from sklearn.metrics import roc_auc_score, classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_val, y_val):
    preds = model.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(y_val, preds)
    print("AUC Score:", auc)

    # convert probabilities → binary
    binary_preds = (preds > 0.5).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_val, binary_preds))

    return auc


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