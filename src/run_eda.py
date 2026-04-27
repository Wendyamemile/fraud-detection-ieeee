from data_loading import load_data
from eda import *

def main():
    df = load_data()

    print("Running EDA...")

    results = {
        "basic_info": basic_info(df),
        "missing_top_20": missing_values(df).head(20),
        "target_distribution": target_distribution(df),
        "fraud_by_hour": fraud_by_hour(df),
        "transaction_amount_stats": transaction_amount_stats(df),
    }

    plot_fraud_by_hour(df)
    save_report(results)

    print("EDA completed. Check outputs/ folder.")


if name == "__main__":
    main()