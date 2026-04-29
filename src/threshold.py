import numpy as np
from sklearn.metrics import f1_score

def find_best_threshold(y_true, probs):
    best_thresh = 0.5
    best_f1 = 0

    for t in np.arange(0.1, 0.9, 0.05):
        preds = (probs > t).astype(int)
        f1 = f1_score(y_true, preds)

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    print(f"Best threshold: {best_thresh}, Best F1: {best_f1}")
    return best_thresh