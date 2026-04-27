import pandas as pd
from config import TARGET, TIME_COL
from feature_engineering import feature_engineering_pipeline


def drop_high_missing(df, threshold=0.9):
    missing_ratio = df.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > threshold].index

    print(f"Dropping {len(cols_to_drop)} high-missing columns")
    return df.drop(columns=cols_to_drop)


def fill_missing(df):
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "float64" or df[col].dtype == "int64":
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("missing")

    return df


def encode_categorical(df):
    df = df.copy()

    cat_cols = df.select_dtypes(include=["object"]).columns

    for col in cat_cols:
        df[col] = df[col].astype("category").cat.codes

    print(f"Encoded {len(cat_cols)} categorical columns")
    return df


def basic_feature_engineering(df):
    df = df.copy()

    # Time feature
    df["hour"] = (df[TIME_COL] / 3600) % 24

    return df


def split_features_target(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def preprocess_pipeline(df):

    df = drop_high_missing(df)
    df = fill_missing(df)
    df = encode_categorical(df)
    # df = basic_feature_engineering(df)
    df = feature_engineering_pipeline(df)

    print("Preprocessing completed")

    return df