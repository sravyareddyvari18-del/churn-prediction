import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# Load test data
X_test = np.load("data/processed/X_test_final.npy")
y_test = np.load("data/processed/y_test.npy")

print("Test data loaded successfully!")
print("X_test shape:", X_test.shape)


# Load models
models = {
    "Logistic Regression": joblib.load(
        "models/logistic_regression.pkl"
    ),
    "Decision Tree": joblib.load(
        "models/decision_tree.pkl"
    ),
    "Random Forest": joblib.load(
        "models/random_forest.pkl"
    )
}


# Store results
results = {}


# Evaluate each model
for name, model in models.items():

    y_pred = model.predict(X_test)

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

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    results[name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))

    print("Confusion Matrix:")
    print(cm)


# Find best model based on F1 score
best_model = max(
    results,
    key=lambda x: results[x]["f1_score"]
)

print("\n" + "=" * 50)
print("BEST MODEL:", best_model)
print("=" * 50)

print(
    "Best F1 Score:",
    round(results[best_model]["f1_score"], 4)
)

print("\nEvaluation completed successfully!")