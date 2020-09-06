import pytest
import numpy as np

from src.data_prep.prep_titanic import load_df, prep_df, split_df, get_feats_and_labels
from src.tree.decision_tree import DecisionTree
from src.tree.random_forest import RandomForest


@pytest.fixture
def dummy_feats_and_labels():
    feats = np.array([[0.7057, -5.4981, 8.3368, -2.8715],
                      [2.4391, 6.4417, -0.80743, -0.69139],
                      [-0.2062, 9.2207, -3.7044, -6.8103],
                      [4.2586, 11.2962, -4.0943, -4.3457],
                      [-2.343, 12.9516, 3.3285, -5.9426],
                      [-2.0545, -10.8679, 9.4926, -1.4116],
                      [2.2279, 4.0951, -4.8037, -2.1112],
                      [-6.1632, 8.7096, -0.21621, -3.6345],
                      [0.52374, 3.644, -4.0746, -1.9909],
                      [1.5077, 1.9596, -3.0584, -0.12243]
                      ])
    labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    return feats, labels


@pytest.fixture
def dummy_titanic():
    df = load_df()
    df = prep_df(df)

    train, test = split_df(df)
    X_train, y_train = get_feats_and_labels(train)
    X_test, y_test = get_feats_and_labels(test)
    return X_train, y_train, X_test, y_test


@pytest.fixture
def dummy_titanic_df():
    df = load_df()
    df.columns = [col.lower() for col in df.columns]

    train, test = split_df(df)
    return train, test


@pytest.fixture
def dummy_passengers():
    # Based on passenger 1 (low passenger class male)
    passenger1 = {'PassengerId': 1,
                  'Survived': None,
                  'Pclass': 3,
                  'Name': ' Mr. Owen',
                  'Sex': 'male',
                  'Age': 22.0,
                  'SibSp': 1,
                  'Parch': 0,
                  'Ticket': 'A/5 21171',
                  'Fare': 7.25,
                  'Cabin': None,
                  'Embarked': 'S'}

    # Based on passenger 2 (high passenger class female)
    passenger2 = {'PassengerId': 2,
                  'Survived': None,
                  'Pclass': 1,
                  'Name': ' Mrs. John',
                  'Sex': 'female',
                  'Age': 38.0,
                  'SibSp': 1,
                  'Parch': 0,
                  'Ticket': 'PC 17599',
                  'Fare': 71.2833,
                  'Cabin': 'C85',
                  'Embarked': 'C'}

    return passenger1, passenger2


@pytest.fixture
def dummy_titanic_dt(dummy_titanic):
    X_train, y_train, _, _ = dummy_titanic
    dt = DecisionTree(depth_limit=5)
    dt.fit(X_train, y_train)
    return dt


@pytest.fixture
def dummy_titanic_rf(dummy_titanic):
    X_train, y_train, _, _ = dummy_titanic
    rf = RandomForest(num_trees=8, depth_limit=5, col_subsampling=0.8, row_subsampling=0.8)
    rf.fit(X_train, y_train)
    return rf
