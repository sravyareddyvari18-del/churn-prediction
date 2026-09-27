import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def run_preprocessing():

    print("Starting Preprocessing Pipeline...")

    # 1. Load dataset
    df = pd.read_csv("data/raw/churn.csv")

    print("Dataset loaded successfully!")
    print("Dataset shape:", df.shape)

    # 2. Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # 3. Handle missing values
    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )

    # 4. Remove customer ID
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    # 5. Convert target to 0/1
    df["Churn"] = df["Churn"].map({
        "Yes": 1,
        "No": 0
    })

    # 6. Separate features and target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # 7. Identify columns
    categorical_cols = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_cols = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    # 8. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 9. Scale numerical features
    scaler = StandardScaler()

    X_train_num = scaler.fit_transform(
        X_train[numerical_cols]
    )

    X_test_num = scaler.transform(
        X_test[numerical_cols]
    )

    # 10. Encode categorical features
    encoder = OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
        sparse_output=False
    )

    X_train_cat = encoder.fit_transform(
        X_train[categorical_cols]
    )

    X_test_cat = encoder.transform(
        X_test[categorical_cols]
    )

    # 11. Combine numerical and categorical features
    X_train_final = np.hstack(
        [X_train_num, X_train_cat]
    )

    X_test_final = np.hstack(
        [X_test_num, X_test_cat]
    )

    # 12. Create required directories
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # 13. Save processed data
    np.save(
        "data/processed/X_train_final.npy",
        X_train_final
    )

    np.save(
        "data/processed/X_test_final.npy",
        X_test_final
    )

    np.save(
        "data/processed/y_train.npy",
        y_train.to_numpy()
    )

    np.save(
        "data/processed/y_test.npy",
        y_test.to_numpy()
    )

    # 14. Save preprocessing objects
    joblib.dump(
        scaler,
        "models/scaler.pkl"
    )

    joblib.dump(
        encoder,
        "models/ohe.pkl"
    )

    # 15. Save metadata
    metadata = {
        "dataset": "Telco Customer Churn",
        "original_shape": list(df.shape),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "numerical_features": numerical_cols,
        "categorical_features": categorical_cols
    }

    with open(
        "data/processed/dataset_metadata.json",
        "w"
    ) as f:
        json.dump(metadata, f, indent=4)

    print("Processed data saved successfully!")
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Preprocessing completed successfully!")


if __name__ == "__main__":
    run_preprocessing()