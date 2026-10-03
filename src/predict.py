"""
predict.py
----------
Utility to predict survival for a new passenger using a trained model.
"""

import numpy as np
import pandas as pd


def encode_passenger(passenger_dict):
    """
    Encode a passenger dictionary into a feature array.

    Expected keys:
        Pclass   : int   (1, 2, or 3)
        Sex      : str   ('male' or 'female')
        Age      : float
        SibSp    : int
        Parch    : int
        Fare     : float
        Embarked : str   ('C', 'Q', or 'S')

    Returns a 2D numpy array ready for model.predict().
    """
    sex_map      = {'male': 0, 'female': 1}
    embarked_map = {'C': 0, 'Q': 1, 'S': 2}

    sex      = sex_map.get(passenger_dict['Sex'].lower(), 0)
    embarked = embarked_map.get(passenger_dict['Embarked'].upper(), 2)

    features = np.array([[
        passenger_dict['Pclass'],
        sex,
        passenger_dict['Age'],
        passenger_dict['SibSp'],
        passenger_dict['Parch'],
        passenger_dict['Fare'],
        embarked
    ]])

    return features


def predict_survival(model, passenger_dict):
    """
    Predict whether a passenger survives.

    Parameters
    ----------
    model          : trained sklearn model
    passenger_dict : dict with passenger details

    Returns
    -------
    prediction : str — 'Survived' or 'Did Not Survive'
    probability: float — confidence of the prediction (if supported)
    """
    features = encode_passenger(passenger_dict)
    prediction = model.predict(features)[0]

    # Get probability if the model supports it
    try:
        prob = model.predict_proba(features)[0][prediction]
    except AttributeError:
        prob = None

    result = "✅ Survived" if prediction == 1 else "❌ Did Not Survive"

    print("\n" + "=" * 45)
    print("  PASSENGER SURVIVAL PREDICTION")
    print("=" * 45)
    print(f"  Passenger Class : {passenger_dict['Pclass']}")
    print(f"  Sex             : {passenger_dict['Sex'].capitalize()}")
    print(f"  Age             : {passenger_dict['Age']}")
    print(f"  SibSp           : {passenger_dict['SibSp']}")
    print(f"  Parch           : {passenger_dict['Parch']}")
    print(f"  Fare            : £{passenger_dict['Fare']:.2f}")
    print(f"  Embarked        : {passenger_dict['Embarked']}")
    print("-" * 45)
    print(f"  Prediction      : {result}")
    if prob is not None:
        print(f"  Confidence      : {prob * 100:.1f}%")
    print("=" * 45)

    return result, prob
