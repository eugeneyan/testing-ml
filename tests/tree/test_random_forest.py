import numpy as np
import pytest
from sklearn.metrics import accuracy_score, roc_auc_score

from src.tree.random_forest import RandomForest, DecisionTree
from tests.tree.test_decision_tree import dummy_titanic, dummy_feats_and_labels
from src.utils.timer import predict_with_time


@pytest.fixture
def dummy_titanic_rf(dummy_titanic):
    X_train, y_train, _, _ = dummy_titanic
    rf = RandomForest(num_trees=7, depth_limit=6, col_subsampling=0.8, row_subsampling=0.8)
    rf.fit(X_train, y_train)
    return rf


# Check if model can overfit perfectly
def test_rf_overfit(dummy_feats_and_labels):
    feats, labels = dummy_feats_and_labels
    dt = RandomForest(1, 1, 1)
    dt.fit(feats, labels)
    pred = np.round(dt.predict(feats))
    assert np.array_equal(labels, pred), 'RandomForest should fit data perfectly with single tree and no subsampling.'


# Check if additional tree depth increases accuracy and AUC ROC
def test_dt_increase_acc(dummy_titanic):
    X_train, y_train, X_test, y_test = dummy_titanic

    acc_list = []
    auc_list = []
    for num_trees in [1, 3, 7, 15]:
        rf = RandomForest(num_trees=num_trees, depth_limit=7, col_subsampling=0.7, row_subsampling=0.7)
        rf.fit(X_train, y_train)
        pred = rf.predict(X_test)
        pred_binary = np.round(pred)
        acc_list.append(accuracy_score(y_test, pred_binary))
        auc_list.append(roc_auc_score(y_test, pred))

    assert sorted(acc_list) == acc_list, 'Accuracy should increase as number of trees increases.'
    assert sorted(auc_list) == auc_list, 'AUC ROC should increase as number of trees increases.'


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


# RandomForest evaluation
def test_dt_evaluation(dummy_titanic_rf, dummy_titanic):
    model = dummy_titanic_rf
    X_train, y_train, X_test, y_test = dummy_titanic

    pred_train = model.predict(X_train)
    pred_train_binary = np.round(pred_train)
    acc_train = accuracy_score(y_train, pred_train_binary)
    auc_train = roc_auc_score(y_train, pred_train)

    assert acc_train > 0.86, 'Accuracy on train should be > 0.85'
    assert auc_train > 0.93, 'AUC ROC on train should be > 0.90'

    pred_test = model.predict(X_test)
    pred_test_binary = np.round(pred_test)
    acc_test = accuracy_score(y_test, pred_test_binary)
    auc_test = roc_auc_score(y_test, pred_test)

    assert acc_test > 0.85, 'Accuracy on test should be > 0.83'
    assert auc_test > 0.86, 'AUC ROC on test should be > 0.86'


def test_dt_latency(dummy_titanic):
    X_train, y_train, X_test, y_test = dummy_titanic

    # Standardize to use depth = 10
    rf = RandomForest(depth_limit=10, num_trees=5, col_subsampling=0.8, row_subsampling=0.8)
    rf.fit(X_train, y_train)

    latency_array = np.array([predict_with_time(rf, X_test)[1] for i in range(500)])
    latency_p99 = np.quantile(latency_array, 0.99)
    assert latency_p99 < 0.018, 'Latency at 99th percentile should be < 0.018 sec'
