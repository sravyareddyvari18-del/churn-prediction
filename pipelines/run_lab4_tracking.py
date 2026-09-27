import os
import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# --------------------------------------------------
# 1. Load processed data
# --------------------------------------------------

X_train = np.load("data/processed/X_train_final.npy")
X_test = np.load("data/processed/X_test_final.npy")

y_train = np.load("data/processed/y_train.npy")
y_test = np.load("data/processed/y_test.npy")

print("Processed data loaded successfully!")
print("Training data:", X_train.shape)
print("Testing data :", X_test.shape)


# --------------------------------------------------
# 2. Create MLflow experiment
# --------------------------------------------------

mlflow.set_experiment("Churn Prediction - Lab 4")


# --------------------------------------------------
# 3. Define models
# --------------------------------------------------

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}


# --------------------------------------------------
# 4. Train and track each model
# --------------------------------------------------

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print("Training:", model_name)
    print("=" * 60)

    with mlflow.start_run(run_name=model_name):

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        # Track parameters
        if model_name == "Logistic Regression":
            mlflow.log_param(
                "max_iter",
                1000
            )

        elif model_name == "Decision Tree":
            mlflow.log_param(
                "random_state",
                42
            )

        elif model_name == "Random Forest":
            mlflow.log_param(
                "n_estimators",
                100
            )

            mlflow.log_param(
                "random_state",
                42
            )

        # Track metrics
        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1_score",
            f1
        )

        # Save model in MLflow
        mlflow.sklearn.log_model(
            model,
            "model"
        )

        print("Accuracy :", round(accuracy, 4))
        print("Precision:", round(precision, 4))
        print("Recall   :", round(recall, 4))
        print("F1 Score :", round(f1, 4))

        print("MLflow run logged successfully!")


print("\n" + "=" * 60)
print("LAB 4 MLflow TRACKING COMPLETED SUCCESSFULLY!")
print("=" * 60)