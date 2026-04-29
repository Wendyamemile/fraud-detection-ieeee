from sklearn.model_selection import train_test_split
from model import get_model
from config import TARGET, MODEL_DIR
import sys, os
import joblib

def train_model(df):
    print("Starting training...")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # ⚠️ temporal split is better, but start simple
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = get_model()

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc"
    )

    print("Training completed")

    return model, X_val, y_val


def save_model(model, name="baseline_model.pkl"):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, name)
    
    joblib.dump(model, path)
    print(f"Model saved at: {path}")