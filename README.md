# Titanic Survival Prediction Using Machine Learning

A beginner-friendly Machine Learning mini project that predicts whether a passenger survived the Titanic disaster based on their personal information.

---

## 📌 Problem Statement

The sinking of the Titanic on April 15, 1912 is one of the most infamous shipwrecks in history. Of the 2,224 passengers and crew aboard, more than 1,500 died. This project aims to build a machine learning model that predicts whether a passenger survived, based on features such as age, gender, passenger class, and fare.

---

## 🎯 Objective

- Analyze the Titanic dataset to understand survival patterns.
- Build and compare classification models (Logistic Regression and Decision Tree).
- Evaluate model performance using standard metrics.
- Create a simple prediction interface for new passenger data.

---

## 📁 Project Structure

```
Titanic-Survival-Prediction/
├── data/
│   └── train.csv                   # Kaggle Titanic training dataset
├── notebooks/
│   └── titanic_prediction.ipynb    # Main Jupyter Notebook
├── src/
│   ├── data_preprocessing.py       # Data cleaning & preprocessing functions
│   ├── model_training.py           # Model training functions
│   └── predict.py                  # Prediction utility
├── outputs/
│   └── (visualizations saved here) # Charts and confusion matrices
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## 📊 Dataset

- **Source:** [Kaggle Titanic Dataset](https://www.kaggle.com/competitions/titanic/data)
- **File Used:** `train.csv`
- **Records:** 891 passengers
- **Target Variable:** `Survived` (0 = Did Not Survive, 1 = Survived)

### Features Used

| Feature   | Description                              |
|-----------|------------------------------------------|
| Pclass    | Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd) |
| Sex       | Gender of the passenger                  |
| Age       | Age of the passenger                     |
| SibSp     | Number of siblings/spouses aboard        |
| Parch     | Number of parents/children aboard        |
| Fare      | Ticket fare paid                         |
| Embarked  | Port of embarkation (C, Q, S)            |

---

## 🔧 Setup & Installation

### 1. Clone or Download the Project

```bash
git clone https://github.com/yourusername/Titanic-Survival-Prediction.git
cd Titanic-Survival-Prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

- Go to [Kaggle Titanic](https://www.kaggle.com/competitions/titanic/data)
- Download `train.csv`
- Place it in the `data/` folder

### 4. Run the Notebook

```bash
jupyter notebook notebooks/titanic_prediction.ipynb
```

---

## 🤖 Models Used

| Model               | Description                                      |
|---------------------|--------------------------------------------------|
| Logistic Regression | Simple linear classifier, good baseline model    |
| Decision Tree       | Tree-based classifier, easy to interpret         |

---

## 📈 Evaluation Metrics

- **Accuracy** – Overall correct predictions
- **Precision** – Of predicted survivors, how many actually survived
- **Recall** – Of actual survivors, how many were correctly predicted
- **F1-Score** – Harmonic mean of Precision and Recall
- **Confusion Matrix** – Visual breakdown of predictions

---

## 🚀 Sample Prediction

The notebook includes a section where you can enter custom passenger details and get a prediction:

```python
sample_passenger = {
    'Pclass': 1,
    'Sex': 'female',
    'Age': 28,
    'SibSp': 0,
    'Parch': 0,
    'Fare': 75.0,
    'Embarked': 'C'
}
# Output: "✅ Survived"
```

---

## 📌 Results Summary

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | ~80%     | ~79%      | ~76%   | ~77%     |
| Decision Tree       | ~78%     | ~76%      | ~74%   | ~75%     |

> Note: Actual values may vary slightly depending on the random split.

---

## 👨‍🎓 About

This project was built as a college mini-project to demonstrate fundamental machine learning concepts including data cleaning, exploratory data analysis, model training, and evaluation.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
