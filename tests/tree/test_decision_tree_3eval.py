import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from src.tree.decision_tree import DecisionTree
from src.utils.timer import fit_with_time, predict_with_time
from tests.tree.fixtures import dummy_titanic_dt, dummy_titanic


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

    assert acc_test > 0.82, 'Accuracy on test should be > 0.82'
    assert auc_test > 0.84, 'AUC ROC on test should be > 0.84'


def test_dt_training_time(dummy_titanic):
    X_train, y_train, X_test, y_test = dummy_titanic

    # Standardize to use depth = 10
    dt = DecisionTree(depth_limit=10)
    latency_array = np.array([fit_with_time(dt, X_train, y_train)[1] for i in range(50)])
    time_p95 = np.quantile(latency_array, 0.95)
    assert time_p95 < 1.0, 'Training time at 95th percentile should be < 1.0 sec'


def test_dt_serving_latency(dummy_titanic):
    X_train, y_train, X_test, y_test = dummy_titanic

    # Standardize to use depth = 10
    dt = DecisionTree(depth_limit=10)
    dt.fit(X_train, y_train)

    latency_array = np.array([predict_with_time(dt, X_test)[1] for i in range(200)])
    latency_p99 = np.quantile(latency_array, 0.99)
    assert latency_p99 < 0.004, 'Serving latency at 99th percentile should be < 0.004 sec'
