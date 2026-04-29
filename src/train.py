from sklearn.model_selection import train_test_split
from model import get_model
from config import TARGET, MODEL_DIR
import sys, os
import joblib

def train_model(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # imbalance ratio
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    print("Scale_pos_weight:", scale_pos_weight)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    model = get_model(scale_pos_weight)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc"
    )

    return model, X_val, y_val

def save_model(model, name="baseline_model.pkl"):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, name)
    
    joblib.dump(model, path)
    print(f"Model saved at: {path}")