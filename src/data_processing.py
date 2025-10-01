# src/data_processing.py
import os
import pandas as pd
import numpy as np

def load_data(path="../data/german_credit_data.csv"):
    df = pd.read_csv(path)
    return df

def create_age_group(df, cutoff=25):
    df = df.copy()
    df['age_group'] = np.where(df['age'] < cutoff, 'young', 'old')
    return df

def train_test_split_and_encode(df, target_col='class', test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    X = df.drop(columns=[target_col])
    y = df[target_col]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)  # bad=0, good=1
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, stratify=y_enc, random_state=random_state
    )

    # One-hot encode (concat to avoid mismatch)
    X_all = pd.get_dummies(pd.concat([X_train, X_test], axis=0), drop_first=True)
    X_train_enc = X_all.iloc[:len(X_train)].copy().reset_index(drop=True)
    X_test_enc = X_all.iloc[len(X_train):].copy().reset_index(drop=True)

    # Ensure data folder exists and save as CSV (robust)
    os.makedirs("../data", exist_ok=True)
    X_train_enc.to_csv("../data/X_train_enc.csv", index=False)
    X_test_enc.to_csv("../data/X_test_enc.csv", index=False)
    pd.Series(y_train).to_csv("../data/y_train.csv", index=False)
    pd.Series(y_test).to_csv("../data/y_test.csv", index=False)

    return X_train_enc, X_test_enc, y_train, y_test, le

def load_preprocessed(data_dir="../data"):
    X_train = pd.read_csv(f"{data_dir}/X_train_enc.csv")
    X_test = pd.read_csv(f"{data_dir}/X_test_enc.csv")
    y_train = pd.read_csv(f"{data_dir}/y_train.csv", header=None).iloc[:,0].values
    y_test = pd.read_csv(f"{data_dir}/y_test.csv", header=None).iloc[:,0].values
    return X_train, X_test, y_train, y_test
