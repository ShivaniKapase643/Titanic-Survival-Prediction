"""
model_training.py
-----------------
Functions for training and evaluating ML models on the Titanic dataset.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import numpy as np


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and testing sets (80:20 ratio)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Training set size : {X_train.shape[0]} samples")
    print(f"Testing set size  : {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train, random_state=42):
    """Train a Logistic Regression model."""
    model = LogisticRegression(max_iter=200, random_state=random_state)
    model.fit(X_train, y_train)
    print("Logistic Regression model trained successfully.")
    return model


def train_decision_tree(X_train, y_train, random_state=42):
    """Train a Decision Tree Classifier."""
    model = DecisionTreeClassifier(max_depth=5, random_state=random_state)
    model.fit(X_train, y_train)
    print("Decision Tree model trained successfully.")
    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate the model and return a dictionary of metrics.
    Prints accuracy, classification report, and confusion matrix.
    """
    y_pred = model.predict(X_test)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    cm        = confusion_matrix(y_test, y_pred)

    print(f"\n{'=' * 50}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'=' * 50}")
    print(f"  Accuracy  : {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Did Not Survive', 'Survived'])}")
    print(f"Confusion Matrix:\n{cm}")

    return {
        'model_name': model_name,
        'accuracy':   accuracy,
        'precision':  precision,
        'recall':     recall,
        'f1_score':   f1,
        'confusion_matrix': cm,
        'y_pred':     y_pred
    }


def compare_models(results_lr, results_dt):
    """Print a side-by-side comparison of both models."""
    print(f"\n{'=' * 60}")
    print("  MODEL COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Metric':<15} {'Logistic Regression':>20} {'Decision Tree':>15}")
    print(f"{'-' * 60}")

    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    labels  = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    for metric, label in zip(metrics, labels):
        lr_val = results_lr[metric]
        dt_val = results_dt[metric]
        print(f"{label:<15} {lr_val:>20.4f} {dt_val:>15.4f}")

    print(f"{'=' * 60}")

    # Determine winner
    if results_lr['accuracy'] >= results_dt['accuracy']:
        print("\n✅ Logistic Regression performs better (or equal) based on Accuracy.")
    else:
        print("\n✅ Decision Tree performs better based on Accuracy.")
