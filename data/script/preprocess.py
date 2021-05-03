import pandas as pd
import numpy as np


def preprocess_web_attack():
    df = pd.read_csv("./raw_data/cicids.csv")

    df.replace([float('inf'), 'Infinity',''], np.nan, inplace=True)
    df = df.dropna()
    df = df.sample(frac=0.05, replace=False, random_state=1)
    df[' Timestamp'] = pd.to_datetime(df[' Timestamp'], infer_datetime_format=True)
    df = df.sort_values(by=[' Timestamp'])

    # drop nan and str columns
    drop_cols = list(df.columns)[0:5]
    drop_cols = list(df.columns)[0:5]
    drop_cols.append(list(df.columns)[6])
    df = df.drop(columns=drop_cols)

    # relabeing and put label in the first column
    df[' Label'] = df[' Label'].map({'BENIGN':"nominal", "Web Attack Brute Force": "anomaly","Web Attack Sql Injection": "anomaly", "Web Attack XSS": "anomaly"})
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    df.to_csv("web_attack.csv", index=False, encoding='utf-8')

def preprocess_creditcard():
    df = pd.read_csv("./raw_data/phpKo8OWT.csv")
    # drop nan and str columns
    df = df.dropna()
    #df = df.drop(columns=['Time'])
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    df['Class'] = df['Class'].map({0:"nominal", 1: "anomaly"})
    df = df.sample(frac=0.025, replace=False, random_state=1)
    df = df.sort_values(by=['Time'])
    df = df.drop(columns=['Time'])
    df.to_csv("creditcard.csv", index=False, encoding='utf-8')



if __name__ == "__main__":
    preprocess_web_attack()
    #preprocess_creditcard()
