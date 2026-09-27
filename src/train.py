import os
import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# Load processed data
X_train = np.load("data/processed/X_train_final.npy")
X_test = np.load("data/processed/X_test_final.npy")

y_train = np.load("data/processed/y_train.npy")
y_test = np.load("data/processed/y_test.npy")

print("Processed data loaded successfully!")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)


# Create models directory
os.makedirs("models", exist_ok=True)


# 1. Logistic Regression
lr = LogisticRegression(max_iter=1000)

lr.fit(X_train, y_train)

joblib.dump(
    lr,
    "models/logistic_regression.pkl"
)

print("Logistic Regression trained successfully!")


# 2. Decision Tree
dt = DecisionTreeClassifier(
    random_state=42
)

dt.fit(X_train, y_train)

joblib.dump(
    dt,
    "models/decision_tree.pkl"
)

print("Decision Tree trained successfully!")


# 3. Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

joblib.dump(
    rf,
    "models/random_forest.pkl"
)

print("Random Forest trained successfully!")


print("\nAll baseline models trained and saved successfully!")