import os
import pandas as pd
import matplotlib.pyplot as plt
from config import OUTPUT_PATH, TARGET, TIME_COL


def basic_info(df):
    return {
        "shape": df.shape,
        "num_columns": len(df.columns),
        "info": df.info(),
        "dtypes": df.dtypes.value_counts().to_dict(),
    }


def missing_values(df):
    return df.isnull().mean().sort_values(ascending=False)


def target_distribution(df):
    return df[TARGET].value_counts(normalize=True)


def create_time_features(df):
    df = df.copy()
    df["hour"] = (df[TIME_COL] / 3600) % 24
    return df


def fraud_by_hour(df):
    df = create_time_features(df)
    return df.groupby("hour")[TARGET].mean()


def transaction_amount_stats(df):
    return df.groupby(TARGET)["TransactionAmt"].describe()


def plot_fraud_by_hour(df):
    df = create_time_features(df)
    fraud_rate = df.groupby("hour")[TARGET].mean()

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    plt.figure()
    fraud_rate.plot()
    plt.title("Fraud Rate by Hour")
    plt.xlabel("Hour")
    plt.ylabel("Fraud Rate")

    plt.savefig(OUTPUT_PATH + "fraud_by_hour.png")
    plt.close()


def save_report(results):
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    with open(OUTPUT_PATH + "eda_report.txt", "w") as f:
        for section, content in results.items():
            f.write(f"\n===== {section.upper()} =====\n")
            f.write(str(content))
            f.write("\n")

def save_plot_fraud_by_hour(df):
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    filepath = os.path.join(OUTPUT_PATH, "fraud_by_hour.png")
    plt.savefig(filepath, bbox_inches="tight") 
    plt.close()