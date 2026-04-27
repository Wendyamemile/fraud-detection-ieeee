import pandas as pd
from config import DATA_PATH

def load_data():

    train = pd.read_csv("../" + DATA_PATH + "train_transaction.csv")
    identity = pd.read_csv("../" + DATA_PATH + "train_identity.csv")

    df = train.merge(identity, on="TransactionID", how="left")

    print(f"Data loaded: {df.shape}")
    return df