import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def load_data(path="data/raw/ds_salaries.csv"):
    """
    Load dataset from given path
    """
    df = pd.read_csv(path)
    return df


def drop_columns(df):
    columns_to_drop = ["salary", "salary_currency", "Unnamed: 0"]
    
    df = df.drop(columns=columns_to_drop)
    return df


def encode_features(df):
    categorical_cols = [
        "experience_level",
        "employment_type",
        "job_title",
        "employee_residence",
        "company_location",
        "company_size"
    ]
    
    encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # 🔥 Save encoders
    os.makedirs("model", exist_ok=True)
    joblib.dump(encoders, "model/encoders.pkl")

    return df

def preprocess_data():
    """
    Full preprocessing pipeline
    """
    df = load_data()
    
    df = drop_columns(df)
    
    df = encode_features(df)
    
    return df


# 🔥 Run this file directly to test
if __name__ == "__main__":
    df = preprocess_data()
    print("Columns:", df.columns)
    
    print("✅ Preprocessing Successful!\n")
    print(df.head())
    print("\nShape:", df.shape)

    