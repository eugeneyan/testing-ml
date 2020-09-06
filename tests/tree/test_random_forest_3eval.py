import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from src.tree.random_forest import RandomForest
from src.utils.timer import predict_with_time, fit_with_time
from tests.tree.test_fixtures import dummy_titanic_rf, dummy_titanic


# RandomForest evaluation
def test_rf_evaluation(dummy_titanic_rf, dummy_titanic):
    model = dummy_titanic_rf
    X_train, y_train, X_test, y_test = dummy_titanic

    pred_train = model.predict(X_train)
    pred_train_binary = np.round(pred_train)
    acc_train = accuracy_score(y_train, pred_train_binary)
    auc_train = roc_auc_score(y_train, pred_train)

    assert acc_train > 0.84, 'Accuracy on train should be > 0.84'
    assert auc_train > 0.91, 'AUC ROC on train should be > 0.90'

    pred_test = model.predict(X_test)
    pred_test_binary = np.round(pred_test)
    acc_test = accuracy_score(y_test, pred_test_binary)
    auc_test = roc_auc_score(y_test, pred_test)

    assert acc_test > 0.84, 'Accuracy on test should be > 0.84'
    assert auc_test > 0.86, 'AUC ROC on test should be > 0.86'


def test_rf_training_time(dummy_titanic):
    X_train, y_train, X_test, y_test = dummy_titanic

    # Standardize to use depth = 10
    rf = RandomForest(depth_limit=10, num_trees=5, col_subsampling=0.8, row_subsampling=0.8)
    latency_array = np.array([fit_with_time(rf, X_train, y_train)[1] for i in range(20)])
    time_p95 = np.quantile(latency_array, 0.95)
    assert time_p95 < 3, 'Training time at 95th percentile should be < 3.0 sec'


def test_rf_serving_latency(dummy_titanic):
    X_train, y_train, X_test, y_test = dummy_titanic

    # Standardize to use depth = 10
    rf = RandomForest(depth_limit=10, num_trees=5, col_subsampling=0.8, row_subsampling=0.8)
    rf.fit(X_train, y_train)

    latency_array = np.array([predict_with_time(rf, X_test)[1] for i in range(200)])
    latency_p99 = np.quantile(latency_array, 0.99)
    assert latency_p99 < 0.018, 'Serving latency at 99th percentile should be < 0.018 sec'
