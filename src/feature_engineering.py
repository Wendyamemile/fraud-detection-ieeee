import pandas as pd
from config import TIME_COL


def sort_by_time(df):
    return df.sort_values(TIME_COL)

def add_time_features(df):
    df = df.copy()
    df["hour"] = (df[TIME_COL] / 3600) % 24
    df["day"] = (df[TIME_COL] / (3600 * 24)) % 7
    return df

def add_time_since_last_txn(df, group_col="card1"):
    df = df.copy()

    df = df.sort_values([group_col, TIME_COL])
    df["time_since_last_txn"] = df.groupby(group_col)[TIME_COL].diff()

    return df

def add_transaction_counts(df, group_col="card1"):
    df = df.copy()

    df = df.sort_values([group_col, TIME_COL])

    # Rolling count in last N transactions (approximation)
    df["txn_count_5"] = df.groupby(group_col).cumcount()
    
    return df

def add_amount_stats(df, group_col="card1"):
    df = df.copy()

    grp = df.groupby(group_col)["TransactionAmt"]

    df["amt_mean"] = grp.transform("mean")
    df["amt_std"] = grp.transform("std")
    df["amt_max"] = grp.transform("max")

    return df

def feature_engineering_pipeline(df):
    print("Starting feature engineering...")

    df = sort_by_time(df)
    df = add_time_features(df)
    df = add_time_since_last_txn(df)
    df = add_transaction_counts(df)
    df = add_amount_stats(df)

    print("Feature engineering completed")

    return df