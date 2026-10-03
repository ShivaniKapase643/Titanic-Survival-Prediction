"""
data_preprocessing.py
---------------------
Functions for loading, cleaning, and preprocessing the Titanic dataset.
"""

import pandas as pd
import numpy as np


def load_data(filepath):
    """Load the Titanic CSV dataset from the given filepath."""
    df = pd.read_csv(filepath)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    return df


def explore_data(df):
    """Print basic exploratory information about the dataset."""
    print("=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)
    print(f"\nShape: {df.shape}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nBasic Statistics:\n{df.describe()}")
    print(f"\nSurvival Distribution:\n{df['Survived'].value_counts()}")


def clean_data(df):
    """
    Clean the dataset:
    - Fill missing Age values with the median age.
    - Fill missing Embarked values with the most frequent value.
    - Drop the Cabin column (too many missing values).
    - Drop unnecessary columns: PassengerId, Name, Ticket.
    """
    df = df.copy()

    # Fill missing Age with median
    df['Age'].fillna(df['Age'].median(), inplace=True)

    # Fill missing Embarked with mode
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    # Drop Cabin column (>77% missing values)
    df.drop(columns=['Cabin'], inplace=True)

    # Drop columns not useful for prediction
    df.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)

    print("Data cleaning complete.")
    print(f"Remaining missing values:\n{df.isnull().sum()}")
    return df


def encode_features(df):
    """
    Encode categorical variables:
    - Sex: male -> 0, female -> 1
    - Embarked: C -> 0, Q -> 1, S -> 2
    """
    df = df.copy()

    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

    print("Categorical encoding complete.")
    return df


def select_features(df):
    """
    Separate features (X) and target variable (y).
    Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
    Target: Survived
    """
    feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = df[feature_cols]
    y = df['Survived']
    print(f"Features selected: {feature_cols}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    return X, y
