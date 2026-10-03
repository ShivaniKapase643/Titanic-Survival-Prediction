"""
Smoke test - verifies the full notebook pipeline runs without errors.
Run from the project root: python smoke_test.py
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ── 1. Load ──────────────────────────────────────────────────
df = pd.read_csv('data/titanic/train.csv')
assert df.shape == (891, 12), f"Unexpected shape: {df.shape}"
print(f"[OK] Dataset loaded: {df.shape}")

# ── 2. Clean ─────────────────────────────────────────────────
df_clean = df.copy()
df_clean['Age'].fillna(df_clean['Age'].median(), inplace=True)
df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0], inplace=True)
df_clean.drop(columns=['Cabin', 'PassengerId', 'Name', 'Ticket'], inplace=True)
assert df_clean.isnull().sum().sum() == 0, "Still has NaN after cleaning"
print(f"[OK] Data cleaned. Shape: {df_clean.shape}")

# ── 3. Encode ────────────────────────────────────────────────
df_encoded = df_clean.copy()
df_encoded['Sex']      = df_encoded['Sex'].map({'male': 0, 'female': 1})
df_encoded['Embarked'] = df_encoded['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
assert df_encoded.isnull().sum().sum() == 0, "NaN introduced during encoding"
print("[OK] Encoding complete.")

# ── 4. Split ─────────────────────────────────────────────────
X = df_encoded[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = df_encoded['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"[OK] Train: {len(X_train)}, Test: {len(X_test)}")

# ── 5. Train ─────────────────────────────────────────────────
lr = LogisticRegression(max_iter=200, random_state=42)
lr.fit(X_train, y_train)
print("[OK] Logistic Regression trained.")

dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
print("[OK] Decision Tree trained.")

# ── 6. Evaluate ──────────────────────────────────────────────
lr_pred = lr.predict(X_test)
dt_pred = dt.predict(X_test)

lr_acc = accuracy_score(y_test, lr_pred)
dt_acc = accuracy_score(y_test, dt_pred)
lr_f1  = f1_score(y_test, lr_pred)
dt_f1  = f1_score(y_test, dt_pred)

assert lr_acc > 0.70, f"LR accuracy too low: {lr_acc}"
assert dt_acc > 0.70, f"DT accuracy too low: {dt_acc}"
print(f"[OK] LR  Accuracy: {lr_acc*100:.2f}%  F1: {lr_f1:.4f}")
print(f"[OK] DT  Accuracy: {dt_acc*100:.2f}%  F1: {dt_f1:.4f}")

# ── 7. Predict sample passengers ─────────────────────────────
# 1st class female — should survive
s1 = np.array([[1, 1, 28, 0, 0, 75.0, 0]])
p1 = lr.predict(s1)[0]
print(f"[OK] 1st class female  -> {'SURVIVED' if p1 == 1 else 'DID NOT SURVIVE'} (expected: SURVIVED)")

# 3rd class male — should not survive
s2 = np.array([[3, 0, 22, 1, 0, 7.25, 2]])
p2 = lr.predict(s2)[0]
print(f"[OK] 3rd class male    -> {'SURVIVED' if p2 == 1 else 'DID NOT SURVIVE'} (expected: DID NOT SURVIVE)")

print()
print("=" * 45)
print("  ALL SMOKE TESTS PASSED - Project is ready!")
print("=" * 45)
