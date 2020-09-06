import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from src.tree.decision_tree import gini_gain, gini_impurity, DecisionTree
from tests.tree.fixtures import dummy_feats_and_labels, dummy_titanic, dummy_titanic_df


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


# Check model prediction to ensure same shape as labels
def test_dt_output_shape(dummy_feats_and_labels, dummy_titanic):
    feats, labels = dummy_feats_and_labels
    dt = DecisionTree()
    dt.fit(feats, labels)
    pred = dt.predict(feats)

    assert pred.shape == (feats.shape[0],), 'DecisionTree output should be same as training labels.'

    X_train, y_train, X_test, y_test = dummy_titanic
    dt = DecisionTree()
    dt.fit(X_train, y_train)
    pred_train = dt.predict(X_train)
    pred_test = dt.predict(X_test)

    assert pred_train.shape == (X_train.shape[0],), 'DecisionTree output should be same as training labels.'
    assert pred_test.shape == (X_test.shape[0],), 'DecisionTree output should be same as testing labels.'


# Check model prediction to ensure output ranges from 0 to 1 inclusive
def test_dt_output_range(dummy_feats_and_labels, dummy_titanic):
    feats, labels = dummy_feats_and_labels
    dt = DecisionTree()
    dt.fit(feats, labels)
    pred = dt.predict(feats)

    assert (pred <= 1).all() & (pred >= 0).all(), 'Decision tree output should range from 0 to 1 inclusive'

    X_train, y_train, X_test, y_test = dummy_titanic
    dt = DecisionTree()
    dt.fit(X_train, y_train)
    pred_train = dt.predict(X_train)
    pred_test = dt.predict(X_test)

    assert (pred_train <= 1).all() & (pred_train >= 0).all(), 'Decision tree output should range from 0 to 1 inclusive'
    assert (pred_test <= 1).all() & (pred_test >= 0).all(), 'Decision tree output should range from 0 to 1 inclusive'


# Check if model can overfit perfectly
def test_dt_overfit(dummy_feats_and_labels, dummy_titanic):
    feats, labels = dummy_feats_and_labels
    dt = DecisionTree()
    dt.fit(feats, labels)
    pred = np.round(dt.predict(feats))

    assert np.array_equal(labels, pred), 'DecisionTree should fit data perfectly and prediction should == labels.'

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

    acc_list, auc_list = [], []
    for depth in range(1, 10):
        dt = DecisionTree(depth_limit=depth)
        dt.fit(X_train, y_train)
        pred = dt.predict(X_train)
        pred_binary = np.round(pred)
        acc_list.append(accuracy_score(y_train, pred_binary))
        auc_list.append(roc_auc_score(y_train, pred))

    assert sorted(acc_list) == acc_list, 'Accuracy should increase as tree depth increases.'
    assert sorted(auc_list) == auc_list, 'AUC ROC should increase as tree depth increases.'


# Check if any records in our test set are also in our train set
def test_data_leak_in_test_data(dummy_titanic_df):
    train, test = dummy_titanic_df

    concat_df = pd.concat([train, test])
    concat_df.drop_duplicates(inplace=True)

    assert concat_df.shape[0] == train.shape[0] + test.shape[0]
