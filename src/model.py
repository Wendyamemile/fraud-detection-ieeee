import lightgbm as lgb

def get_model(scale_pos_weight=None):
    params = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 64,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    }

    # optional improvement for imbalance
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight

    return lgb.LGBMClassifier(**params)