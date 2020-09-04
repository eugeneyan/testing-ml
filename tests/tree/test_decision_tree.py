import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, roc_auc_score

from src.data_prep.prep_titanic import load_df, prep_df, split_df, get_feats_and_labels
from src.tree.decision_tree import gini_gain, gini_impurity, DecisionTree
from src.utils.timer import predict_with_time, train_with_time


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
def dummy_passengers():
    # Based on passenger 1 (low passenger class male)
    passenger1 = {'PassengerId': 1,
                  'Survived': -1,
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
                  'Survived': -1,
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


def test_gini_impurity():
    assert round(gini_impurity([1, 1, 1, 1, 1, 1, 1, 1]), 3) == 0
    assert round(gini_impurity([1, 1, 1, 1, 1, 1, 1, 0]), 3) == 0.219
    assert round(gini_impurity([1, 1, 1, 1, 1, 1, 0, 0]), 3) == 0.375
    assert round(gini_impurity([1, 1, 1, 1, 1, 0, 0, 0]), 3) == 0.469
    assert round(gini_impurity([1, 1, 1, 1, 0, 0, 0, 0]), 3) == 0.500
    assert round(gini_impurity([1, 1, 0, 0, 0, 0, 0, 0]), 3) == 0.375


def test_gini_gain():
    assert round(gini_gain([1, 1, 1, 1, 0, 0, 0, 0], [[1, 1, 1, 1], [0, 0, 0, 0]]), 3) == 0.5
    assert round(gini_gain([1, 1, 1, 1, 0, 0, 0, 0], [[1, 1, 1, 0], [0, 0, 0, 1]]), 3) == 0.125
    assert round(gini_gain([1, 1, 1, 1, 0, 0, 0, 0], [[1, 0, 0, 0], [0, 1, 1, 1]]), 3) == 0.125
    assert round(gini_gain([1, 1, 1, 1, 0, 0, 0, 0], [[1, 1, 0, 0], [0, 0, 1, 1]]), 3) == 0.0


# Check model prediction to ensure: (i) same shape as labels, (ii) ranges from 0 to 1 inclusive
def test_dt_output(dummy_feats_and_labels, dummy_titanic):
    feats, labels = dummy_feats_and_labels
    dt = DecisionTree()
    dt.fit(feats, labels)
    pred = dt.predict(feats)

    assert pred.shape == (10,), 'DecisionTree output should be same as training labels.'
    assert (pred <= 1).all() & (pred >= 0).all()

    X_train, y_train, X_test, y_test = dummy_titanic
    dt = DecisionTree()
    dt.fit(X_train, y_train)

    pred_train = dt.predict(X_train)
    pred_test = dt.predict(X_test)
    assert pred_train.shape == (712,), 'DecisionTree output should be same as training labels.'
    assert pred_test.shape == (179,), 'DecisionTree output should be same as testing labels.'

    assert (pred_train <= 1).all() & (pred_train >= 0).all()
    assert (pred_test <= 1).all() & (pred_test >= 0).all()


# Check if model can overfit perfectly
def test_dt_overfit(dummy_feats_and_labels, dummy_titanic):
    feats, labels = dummy_feats_and_labels
    dt = DecisionTree()
    dt.fit(feats, labels)
    pred = np.round(dt.predict(feats))

    assert np.array_equal(labels, pred), 'DecisionTree should fit data perfectly.'

    X_train, y_train, X_test, y_test = dummy_titanic
    dt = DecisionTree()
    dt.fit(X_train, y_train)

    pred_train = dt.predict(X_train)
    pred_train_binary = np.round(pred_train)
    acc_train = accuracy_score(y_train, pred_train_binary)
    auc_train = roc_auc_score(y_train, pred_train)

    assert acc_train > 0.97, 'Accuracy on train should be > 0.97'
    assert auc_train > 0.99, 'AUC ROC on train should be > 0.99'


# Check if additional tree depth increases accuracy and AUC ROC
def test_dt_increase_acc(dummy_titanic):
    X_train, y_train, _, _ = dummy_titanic

    acc_list = []
    auc_list = []
    for depth in range(1, 10):
        dt = DecisionTree(depth_limit=depth)
        dt.fit(X_train, y_train)
        pred = dt.predict(X_train)
        pred_binary = np.round(pred)
        acc_list.append(accuracy_score(y_train, pred_binary))
        auc_list.append(roc_auc_score(y_train, pred))

    assert sorted(acc_list) == acc_list, 'Accuracy should increase as tree depth increases.'
    assert sorted(auc_list) == auc_list, 'AUC ROC should increase as tree depth increases.'


# Check if changing certain inputs will keep outputs constant
def test_dt_invariance(dummy_titanic_dt, dummy_passengers):
    model = dummy_titanic_dt
    p1, p2 = dummy_passengers

    # Get original survival probability of passenger 1
    test_df = pd.DataFrame.from_dict([p1], orient='columns')
    X, y = get_feats_and_labels(prep_df(test_df))
    p1_prob = model.predict(X)[0]

    # Change name from Owen to Mary (without changing gender or title)
    p1_name = p1.copy()
    p1_name['Name'] = ' Mr. Mary'
    test_df = pd.DataFrame.from_dict([p1_name], orient='columns')
    X, y = get_feats_and_labels(prep_df(test_df))
    p1_name_prob = model.predict(X)[0]

    # Change ticket number from 'A/5 21171' to 'PC 17599'
    p1_ticket = p1.copy()
    p1_ticket['ticket'] = 'PC 17599'
    test_df = pd.DataFrame.from_dict([p1_ticket], orient='columns')
    X, y = get_feats_and_labels(prep_df(test_df))
    p1_ticket_prob = model.predict(X)[0]

    # Change embarked port from 'S' to 'C'
    p1_port = p1.copy()
    p1_port['Embarked'] = 'C'
    test_df = pd.DataFrame.from_dict([p1_port], orient='columns')
    X, y = get_feats_and_labels(prep_df(test_df))
    p1_port_prob = model.predict(X)[0]

    # Get original survival probability of passenger 2
    test_df = pd.DataFrame.from_dict([p2], orient='columns')
    X, y = get_feats_and_labels(prep_df(test_df))
    p2_prob = model.predict(X)[0]

    # Change name from John to Berns (without changing gender or title)
    p2_name = p2.copy()
    p2_name['Name'] = ' Mrs. Berns'
    test_df = pd.DataFrame.from_dict([p2_name], orient='columns')
    X, y = get_feats_and_labels(prep_df(test_df))
    p2_name_prob = model.predict(X)[0]

    # Change ticket number from 'PC 17599' to 'A/5 21171'
    p2_ticket = p2.copy()
    p2_ticket['ticket'] = 'A/5 21171'
    test_df = pd.DataFrame.from_dict([p2_ticket], orient='columns')
    X, y = get_feats_and_labels(prep_df(test_df))
    p2_ticket_prob = model.predict(X)[0]

    # Change embarked port from 'C' to 'S'
    p2_port = p2.copy()
    p2_port['Embarked'] = 'S'
    test_df = pd.DataFrame.from_dict([p2_port], orient='columns')
    X, y = get_feats_and_labels(prep_df(test_df))
    p2_port_prob = model.predict(X)[0]

    assert p1_prob == p1_name_prob == p1_ticket_prob == p1_port_prob
    assert p2_prob == p2_name_prob == p2_ticket_prob == p2_port_prob


# Check if changing input (e.g., gender, passenger class) will affect survival probability in expected direction
def test_dt_directional_expectation(dummy_titanic_dt, dummy_passengers):
    model = dummy_titanic_dt
    p1, p2 = dummy_passengers

    # Get original survival probability of passenger 1
    test_df = pd.DataFrame.from_dict([p1], orient='columns')
    X, y = get_feats_and_labels(prep_df(test_df))
    p1_prob = model.predict(X)[0]

    # Change gender from male to female
    p1_female = p1.copy()
    p1_female['Name'] = ' Mrs. Owen'
    p1_female['Sex'] = 'female'
    test_df = pd.DataFrame.from_dict([p1_female], orient='columns')
    X, y = get_feats_and_labels(prep_df(test_df))
    p1_female_prob = model.predict(X)[0]

    # Change passenger class from 3 to 1
    p1_class = p1.copy()
    p1_class['Pclass'] = 1
    test_df = pd.DataFrame.from_dict([p1_class], orient='columns')
    X, y = get_feats_and_labels(prep_df(test_df))
    p1_class_prob = model.predict(X)[0]

    assert p1_prob < p1_female_prob, 'Changing gender from male to female should increase survival probability.'
    assert p1_prob < p1_class_prob, 'Changing class from 3 to 1 should increase survival probability.'

    # Get original survival probability of passenger 2
    test_df = pd.DataFrame.from_dict([p2], orient='columns')
    X, y = get_feats_and_labels(prep_df(test_df))
    p2_prob = model.predict(X)[0]

    # Change gender from female to male
    p2_male = p2.copy()
    p2_male['Name'] = ' Mr. John'
    p2_male['Sex'] = 'male'
    test_df = pd.DataFrame.from_dict([p2_male], orient='columns')
    X, y = get_feats_and_labels(prep_df(test_df))
    p2_male_prob = model.predict(X)[0]

    # Change class from 1 to 3
    p2_class = p2.copy()
    p2_class['Pclass'] = 3
    test_df = pd.DataFrame.from_dict([p2_class], orient='columns')
    X, y = get_feats_and_labels(prep_df(test_df))
    p2_class_prob = model.predict(X)[0]

    # Lower fare from 71.2833 to 5
    p2_fare = p2.copy()
    p2_fare['Fare'] = 5
    test_df = pd.DataFrame.from_dict([p2_fare], orient='columns')
    X, y = get_feats_and_labels(prep_df(test_df))
    p2_fare_prob = model.predict(X)[0]

    assert p2_prob > p2_male_prob, 'Changing gender from female to male should decrease survival probability.'
    assert p2_prob > p2_class_prob, 'Changing class from 1 to 3 should decrease survival probability.'
    assert p2_prob > p2_fare_prob, 'Changing fare from 72 to 5 should decrease survival probability.'


# Check for minimum functionality (e.g., missing values)
# Note: This is a test of data_prep as the missing values handling is done in data_prep
def test_dt_functionality(dummy_titanic_dt, dummy_passengers):
    model = dummy_titanic_dt
    p1, p1 = dummy_passengers

    # Set numeric cols to null
    p1_num_null = p1.copy()
    p1_num_null['Pclass'] = None
    p1_num_null['Age'] = None
    p1_num_null['SibSp'] = None
    p1_num_null['Parch'] = None
    p1_num_null['Fare'] = None
    test_df = pd.DataFrame.from_dict([p1_num_null], orient='columns')
    X, y = get_feats_and_labels(prep_df(test_df))
    p1_num_null_prob = model.predict(X)[0]

    p2_num_null = p1.copy()
    p2_num_null['Pclass'] = None
    p2_num_null['Age'] = None
    p2_num_null['SibSp'] = None
    p2_num_null['Parch'] = None
    p2_num_null['Fare'] = None
    test_df = pd.DataFrame.from_dict([p2_num_null], orient='columns')
    X, y = get_feats_and_labels(prep_df(test_df))
    p2_num_null_prob = model.predict(X)[0]

    # Set categorical cols to null
    p1_cat_null = p1.copy()
    p1_cat_null['Name'] = None
    p1_cat_null['Sex'] = None
    p1_cat_null['Ticket'] = None
    test_df = pd.DataFrame.from_dict([p1_cat_null], orient='columns')
    X, y = get_feats_and_labels(prep_df(test_df))
    p1_cat_null_prob = model.predict(X)[0]

    p2_cat_null = p1.copy()
    p2_cat_null['Name'] = None
    p2_cat_null['Sex'] = None
    p2_cat_null['Ticket'] = None
    test_df = pd.DataFrame.from_dict([p2_cat_null], orient='columns')
    X, y = get_feats_and_labels(prep_df(test_df))
    p2_cat_null_prob = model.predict(X)[0]

    assert p1_num_null_prob
    assert p2_num_null_prob
    assert p1_cat_null_prob
    assert p2_cat_null_prob


# DecisionTree evaluation
def test_dt_evaluation(dummy_titanic_dt, dummy_titanic):
    model = dummy_titanic_dt
    X_train, y_train, X_test, y_test = dummy_titanic

    pred_train = model.predict(X_train)
    pred_train_binary = np.round(pred_train)
    acc_train = accuracy_score(y_train, pred_train_binary)
    auc_train = roc_auc_score(y_train, pred_train)

    assert acc_train > 0.85, 'Accuracy on train should be > 0.85'
    assert auc_train > 0.90, 'AUC ROC on train should be > 0.90'

    pred_test = model.predict(X_test)
    pred_test_binary = np.round(pred_test)
    acc_test = accuracy_score(y_test, pred_test_binary)
    auc_test = roc_auc_score(y_test, pred_test)

    assert acc_test > 0.83, 'Accuracy on test should be > 0.83'
    assert auc_test > 0.84, 'AUC ROC on test should be > 0.84'


def test_dt_training_time(dummy_titanic):
    X_train, y_train, X_test, y_test = dummy_titanic

    # Standardize to use depth = 10
    dt = DecisionTree(depth_limit=10)
    latency_array = np.array([train_with_time(dt, X_train, y_train)[1] for i in range(100)])
    time_p95 = np.quantile(latency_array, 0.95)
    assert time_p95 < 1.0, 'Training time at 95th percentile should be < 1.0 sec'


def test_dt_serving_latency(dummy_titanic):
    X_train, y_train, X_test, y_test = dummy_titanic

    # Standardize to use depth = 10
    dt = DecisionTree(depth_limit=10)
    dt.fit(X_train, y_train)

    latency_array = np.array([predict_with_time(dt, X_test)[1] for i in range(500)])
    latency_p99 = np.quantile(latency_array, 0.99)
    assert latency_p99 < 0.004, 'Serving latency at 99th percentile should be < 0.004 sec'
