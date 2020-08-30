import numpy as np
import pytest
from sklearn.metrics import accuracy_score, roc_auc_score

from src.tree.random_forest import RandomForest, DecisionTree
from tests.tree.test_decision_tree import dummy_titanic, dummy_feats_and_labels


@pytest.fixture
def dummy_titanic_rf(dummy_titanic):
    X_train, y_train, _, _ = dummy_titanic
    rf = RandomForest(depth_limit=5, num_trees=7, col_subsampling=0.8, row_subsampling=0.8)
    rf.fit(X_train, y_train)
    return rf


# Check if model can overfit perfectly
def test_rf_overfit(dummy_feats_and_labels):
    feats, labels = dummy_feats_and_labels
    dt = RandomForest(1, 1, 1)
    dt.fit(feats, labels)
    pred = np.round(dt.predict(feats))
    assert np.array_equal(labels, pred), 'RandomForest should fit data perfectly with single tree and no subsampling.'


# Check if RandomForest is an improvement over DecisionTree on test set
def test_rf_better_than_dt(dummy_titanic):
    X_train, y_train, X_test, y_test = dummy_titanic

    dt = DecisionTree(depth_limit=10)
    dt.fit(X_train, y_train)

    rf = RandomForest(depth_limit=10, num_trees=7, col_subsampling=0.8, row_subsampling=0.8)
    rf.fit(X_train, y_train)

    pred_test_dt = dt.predict(X_test)
    pred_test_binary_dt = np.round(pred_test_dt)
    acc_test_dt = accuracy_score(y_test, pred_test_binary_dt)
    auc_test_dt = roc_auc_score(y_test, pred_test_dt)

    pred_test_rf = rf.predict(X_test)
    pred_test_binary_rf = np.round(pred_test_rf)
    acc_test_rf = accuracy_score(y_test, pred_test_binary_rf)
    auc_test_rf = roc_auc_score(y_test, pred_test_rf)

    assert acc_test_rf > acc_test_dt, 'RandomForest should have higher accuracy than DecisionTree on test set.'
    assert auc_test_rf > auc_test_dt, 'RandomForest should have higher AUC ROC than DecisionTree on test set.'
