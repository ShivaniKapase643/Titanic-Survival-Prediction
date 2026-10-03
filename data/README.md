# Data Folder

Place the Kaggle Titanic dataset file here.

## How to Download

1. Go to: https://www.kaggle.com/competitions/titanic/data
2. Sign in to your Kaggle account (free)
3. Download `train.csv`
4. Place `train.csv` in this `data/` folder

## Expected File

```
data/
└── train.csv   ← place this file here
```

## Dataset Description

| Column      | Description                                         |
|-------------|-----------------------------------------------------|
| PassengerId | Unique ID for each passenger                        |
| Survived    | 0 = Did not survive, 1 = Survived (Target variable) |
| Pclass      | Passenger class: 1 = 1st, 2 = 2nd, 3 = 3rd         |
| Name        | Passenger name                                      |
| Sex         | male / female                                       |
| Age         | Age in years                                        |
| SibSp       | Number of siblings/spouses aboard                   |
| Parch       | Number of parents/children aboard                   |
| Ticket      | Ticket number                                       |
| Fare        | Ticket fare                                         |
| Cabin       | Cabin number (many missing values)                  |
| Embarked    | Port of embarkation: C = Cherbourg, Q = Queenstown, S = Southampton |
