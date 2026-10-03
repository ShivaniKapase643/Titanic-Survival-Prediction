"""
app.py - Titanic Survival Prediction Web App
Run locally : streamlit run app.py
Deploy to   : https://streamlit.io/cloud
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon=":ship:",
    layout="wide"
)

# ─────────────────────────────────────────────
# Load & train (cached so it runs only once)
# ─────────────────────────────────────────────
@st.cache_data
def load_and_train():
    df = pd.read_csv('data/titanic/train.csv')

    df_clean = df.copy()
    df_clean['Age'].fillna(df_clean['Age'].median(), inplace=True)
    df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0], inplace=True)
    df_clean.drop(columns=['Cabin', 'PassengerId', 'Name', 'Ticket'], inplace=True)

    df_encoded = df_clean.copy()
    df_encoded['Sex']      = df_encoded['Sex'].map({'male': 0, 'female': 1})
    df_encoded['Embarked'] = df_encoded['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

    X = df_encoded[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    y = df_encoded['Survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lr = LogisticRegression(max_iter=200, random_state=42)
    lr.fit(X_train, y_train)

    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)

    return df_clean, df_encoded, lr, dt, X_test, y_test


df_clean, df_encoded, lr_model, dt_model, X_test, y_test = load_and_train()

lr_pred = lr_model.predict(X_test)
dt_pred = dt_model.predict(X_test)

# ─────────────────────────────────────────────
# Sidebar — navigation
# ─────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Dataset Overview", "EDA Charts", "Model Results", "Predict a Passenger"]
)

# ═══════════════════════════════════════════════
# PAGE: Home
# ═══════════════════════════════════════════════
if page == "Home":
    st.title("Titanic Survival Prediction")
    st.subheader("Using Machine Learning — Logistic Regression & Decision Tree")

    st.markdown("""
    ---
    ### Problem Statement
    On **April 15, 1912**, the RMS Titanic sank after colliding with an iceberg.
    Of the 2,224 passengers and crew, more than **1,500 died**.

    This app uses Machine Learning to predict whether a passenger would have **survived**
    based on personal details like age, gender, passenger class, and fare paid.

    ---
    ### Project Summary
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Passengers", "891")
    col2.metric("Survival Rate",    f"{df_clean['Survived'].mean()*100:.1f}%")
    col3.metric("Features Used",    "7")
    col4.metric("Models Trained",   "2")

    st.markdown("""
    ---
    ### How to Use
    - Use the **sidebar** to navigate between sections.
    - Visit **EDA Charts** to explore survival patterns.
    - Visit **Model Results** to see accuracy and confusion matrices.
    - Visit **Predict a Passenger** to enter passenger details and get a prediction.
    """)

# ═══════════════════════════════════════════════
# PAGE: Dataset Overview
# ═══════════════════════════════════════════════
elif page == "Dataset Overview":
    st.title("Dataset Overview")

    st.subheader("First 10 rows")
    st.dataframe(df_clean.head(10), use_container_width=True)

    st.subheader("Dataset Shape")
    st.write(f"Rows: **{df_clean.shape[0]}**  |  Columns: **{df_clean.shape[1]}**")

    st.subheader("Data Types & Missing Values")
    info_df = pd.DataFrame({
        'Column'       : df_clean.columns,
        'Data Type'    : df_clean.dtypes.values,
        'Missing Count': df_clean.isnull().sum().values,
        'Missing %'    : (df_clean.isnull().sum().values / len(df_clean) * 100).round(2)
    })
    st.dataframe(info_df, use_container_width=True)

    st.subheader("Statistical Summary")
    st.dataframe(df_clean.describe(), use_container_width=True)

    st.subheader("Survival Distribution")
    vc = df_clean['Survived'].value_counts()
    st.write(f"- Did Not Survive (0): **{vc[0]}** passengers")
    st.write(f"- Survived (1)       : **{vc[1]}** passengers")

# ═══════════════════════════════════════════════
# PAGE: EDA Charts
# ═══════════════════════════════════════════════
elif page == "EDA Charts":
    st.title("Exploratory Data Analysis")

    chart = st.selectbox(
        "Select a chart:",
        ["Survival Count", "Survival by Gender", "Survival by Class",
         "Age Distribution", "Fare Distribution", "Correlation Heatmap"]
    )

    # ── Survival Count ──────────────────────────
    if chart == "Survival Count":
        st.subheader("Survival Count")
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = df_clean['Survived'].value_counts()
        bars = ax.bar(['Did Not Survive', 'Survived'], counts.values,
                      color=['#E74C3C', '#2ECC71'], edgecolor='black', width=0.5)
        for bar, count in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 4,
                    str(count), ha='center', fontsize=12, fontweight='bold')
        ax.set_title('Survival Count', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Passengers')
        ax.set_ylim(0, max(counts.values) + 60)
        st.pyplot(fig)
        st.info(f"Only **{df_clean['Survived'].mean()*100:.1f}%** of passengers survived.")

    # ── Survival by Gender ───────────────────────
    elif chart == "Survival by Gender":
        st.subheader("Survival by Gender")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        sns.countplot(data=df_clean, x='Sex', hue='Survived',
                      palette={0: '#E74C3C', 1: '#2ECC71'}, ax=axes[0], edgecolor='black')
        axes[0].set_title('Count by Gender')
        axes[0].set_ylabel('Number of Passengers')

        sr = df_clean.groupby('Sex')['Survived'].mean() * 100
        sr.plot(kind='bar', color=['#3498DB', '#E91E63'], edgecolor='black', ax=axes[1], rot=0)
        axes[1].set_title('Survival Rate by Gender (%)')
        axes[1].set_ylabel('Survival Rate (%)')
        axes[1].set_ylim(0, 100)
        for i, v in enumerate(sr):
            axes[1].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        st.info("Females had a **~74%** survival rate vs **~19%** for males.")

    # ── Survival by Class ────────────────────────
    elif chart == "Survival by Class":
        st.subheader("Survival by Passenger Class")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        sns.countplot(data=df_clean, x='Pclass', hue='Survived',
                      palette={0: '#E74C3C', 1: '#2ECC71'}, ax=axes[0], edgecolor='black')
        axes[0].set_title('Count by Class')
        axes[0].set_ylabel('Number of Passengers')

        cs = df_clean.groupby('Pclass')['Survived'].mean() * 100
        cs.plot(kind='bar', color=['#1ABC9C', '#F39C12', '#E74C3C'],
                edgecolor='black', ax=axes[1], rot=0)
        axes[1].set_title('Survival Rate by Class (%)')
        axes[1].set_ylabel('Survival Rate (%)')
        axes[1].set_ylim(0, 100)
        axes[1].set_xticklabels(['1st', '2nd', '3rd'])
        for i, v in enumerate(cs):
            axes[1].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        st.info("1st class passengers had the highest survival rate (~63%).")

    # ── Age Distribution ─────────────────────────
    elif chart == "Age Distribution":
        st.subheader("Age Distribution")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(df_clean['Age'], bins=30, color='#3498DB', edgecolor='black', alpha=0.8)
        axes[0].axvline(df_clean['Age'].mean(), color='red', linestyle='--',
                        label=f'Mean: {df_clean["Age"].mean():.1f}')
        axes[0].axvline(df_clean['Age'].median(), color='orange', linestyle='--',
                        label=f'Median: {df_clean["Age"].median():.1f}')
        axes[0].set_title('Overall Age Distribution')
        axes[0].legend()

        axes[1].hist(df_clean[df_clean['Survived']==0]['Age'], bins=25,
                     alpha=0.7, color='#E74C3C', edgecolor='black', label='Did Not Survive')
        axes[1].hist(df_clean[df_clean['Survived']==1]['Age'], bins=25,
                     alpha=0.7, color='#2ECC71', edgecolor='black', label='Survived')
        axes[1].set_title('Age by Survival')
        axes[1].legend()
        plt.tight_layout()
        st.pyplot(fig)

    # ── Fare Distribution ────────────────────────
    elif chart == "Fare Distribution":
        st.subheader("Fare Distribution")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(df_clean['Fare'], bins=40, color='#9B59B6', edgecolor='black', alpha=0.8)
        axes[0].axvline(df_clean['Fare'].mean(), color='red', linestyle='--',
                        label=f'Mean: {df_clean["Fare"].mean():.1f}')
        axes[0].set_title('Fare Distribution')
        axes[0].legend()

        df_clean.boxplot(column='Fare', by='Survived', ax=axes[1])
        axes[1].set_title('Fare by Survival')
        axes[1].set_xlabel('Survived (0=No, 1=Yes)')
        plt.suptitle('')
        plt.tight_layout()
        st.pyplot(fig)

    # ── Correlation Heatmap ──────────────────────
    elif chart == "Correlation Heatmap":
        st.subheader("Feature Correlation Heatmap")
        df_corr = df_clean.copy()
        df_corr['Sex']      = df_corr['Sex'].map({'male': 0, 'female': 1})
        df_corr['Embarked'] = df_corr['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
        fig, ax = plt.subplots(figsize=(8, 6))
        mask = np.triu(np.ones_like(df_corr.corr(), dtype=bool))
        sns.heatmap(df_corr.corr(), mask=mask, annot=True, fmt='.2f',
                    cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
        ax.set_title('Correlation Heatmap', fontsize=13, fontweight='bold')
        st.pyplot(fig)
        st.info("Sex (gender) has the strongest correlation with survival.")

# ═══════════════════════════════════════════════
# PAGE: Model Results
# ═══════════════════════════════════════════════
elif page == "Model Results":
    st.title("Model Evaluation Results")

    # Metrics table
    metrics_data = {
        'Metric'             : ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Logistic Regression': [
            f"{accuracy_score(y_test, lr_pred)*100:.2f}%",
            f"{precision_score(y_test, lr_pred):.4f}",
            f"{recall_score(y_test, lr_pred):.4f}",
            f"{f1_score(y_test, lr_pred):.4f}"
        ],
        'Decision Tree': [
            f"{accuracy_score(y_test, dt_pred)*100:.2f}%",
            f"{precision_score(y_test, dt_pred):.4f}",
            f"{recall_score(y_test, dt_pred):.4f}",
            f"{f1_score(y_test, dt_pred):.4f}"
        ]
    }
    st.subheader("Performance Metrics")
    st.dataframe(pd.DataFrame(metrics_data).set_index('Metric'), use_container_width=True)

    # Confusion matrices
    st.subheader("Confusion Matrices")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    class_labels = ['Did Not Survive', 'Survived']

    sns.heatmap(confusion_matrix(y_test, lr_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels, ax=axes[0], cbar=False)
    axes[0].set_title('Logistic Regression', fontweight='bold')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    sns.heatmap(confusion_matrix(y_test, dt_pred), annot=True, fmt='d', cmap='Greens',
                xticklabels=class_labels, yticklabels=class_labels, ax=axes[1], cbar=False)
    axes[1].set_title('Decision Tree', fontweight='bold')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')

    plt.tight_layout()
    st.pyplot(fig)

    # Bar comparison chart
    st.subheader("Side-by-Side Comparison")
    metrics  = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    lr_vals  = [accuracy_score(y_test, lr_pred), precision_score(y_test, lr_pred),
                recall_score(y_test, lr_pred),    f1_score(y_test, lr_pred)]
    dt_vals  = [accuracy_score(y_test, dt_pred), precision_score(y_test, dt_pred),
                recall_score(y_test, dt_pred),    f1_score(y_test, dt_pred)]

    x     = np.arange(len(metrics))
    width = 0.35
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    b1 = ax2.bar(x - width/2, lr_vals, width, label='Logistic Regression',
                 color='#3498DB', edgecolor='black', alpha=0.9)
    b2 = ax2.bar(x + width/2, dt_vals, width, label='Decision Tree',
                 color='#2ECC71', edgecolor='black', alpha=0.9)
    for b in list(b1) + list(b2):
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                 f'{b.get_height():.3f}', ha='center', fontsize=9, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel('Score')
    ax2.set_title('Logistic Regression vs Decision Tree', fontweight='bold')
    ax2.legend()
    ax2.axhline(0.8, color='red', linestyle='--', alpha=0.4, linewidth=1.2)
    st.pyplot(fig2)

# ═══════════════════════════════════════════════
# PAGE: Predict a Passenger
# ═══════════════════════════════════════════════
elif page == "Predict a Passenger":
    st.title("Predict Passenger Survival")
    st.markdown("Enter the passenger details below and click **Predict**.")

    col1, col2 = st.columns(2)

    with col1:
        pclass   = st.selectbox("Passenger Class", [1, 2, 3],
                                format_func=lambda x: f"{x}st Class" if x==1
                                else (f"{x}nd Class" if x==2 else f"{x}rd Class"))
        sex      = st.radio("Sex", ["male", "female"])
        age      = st.slider("Age", min_value=1, max_value=80, value=28)
        embarked = st.selectbox("Port of Embarkation",
                                ["S", "C", "Q"],
                                format_func=lambda x: {"S": "S - Southampton",
                                                        "C": "C - Cherbourg",
                                                        "Q": "Q - Queenstown"}[x])

    with col2:
        sibsp  = st.number_input("Siblings / Spouses aboard", min_value=0, max_value=8, value=0)
        parch  = st.number_input("Parents / Children aboard", min_value=0, max_value=6, value=0)
        fare   = st.number_input("Fare paid", min_value=0.0, max_value=600.0, value=30.0, step=0.5)
        model_choice = st.selectbox("Choose Model", ["Logistic Regression", "Decision Tree"])

    if st.button("Predict Survival", use_container_width=True):
        sex_enc      = 0 if sex == 'male' else 1
        embarked_enc = {'C': 0, 'Q': 1, 'S': 2}[embarked]
        features     = np.array([[pclass, sex_enc, age, sibsp, parch, fare, embarked_enc]])

        model = lr_model if model_choice == "Logistic Regression" else dt_model
        pred  = model.predict(features)[0]
        prob  = model.predict_proba(features)[0][pred]

        st.markdown("---")
        if pred == 1:
            st.success(f"### SURVIVED")
            st.markdown(f"**Confidence:** {prob*100:.1f}%")
        else:
            st.error(f"### DID NOT SURVIVE")
            st.markdown(f"**Confidence:** {prob*100:.1f}%")

        # Show input summary
        st.markdown("**Passenger Summary:**")
        summary = pd.DataFrame({
            'Feature' : ['Class', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
            'Value'   : [pclass, sex, age, sibsp, parch, fare, embarked]
        })
        st.dataframe(summary.set_index('Feature'), use_container_width=True)
