"""
Logger utility
"""
from functools import wraps
from time import perf_counter
from typing import Callable
from typing import Tuple

import numpy as np


def timer(func: Callable) -> Callable:
    """Decorator to time a function.

    Args:
        func: Function to time

    Returns:
        Function results and time (in seconds)
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        results = func(*args, **kwargs)
        end = perf_counter()
        run_time = end - start
        return results, run_time

    return wrapper


@timer
def fit_with_time(model, X_train: np.array, y_train: np.array) -> Tuple:
    """Returns trained model with the time

    Args:
        model: Model to test latency on
        X_test: Input data

    Returns:
        Predicted values and time taken to predict it
    """
    return model.fit(X_train, y_train)


@timer
def predict_with_time(model, X_test: np.array) -> Tuple[np.array]:
    """Returns model output with the time

    Args:
        model: Model to test latency on
        X_test: Input data

    Returns:
        Predicted values and time taken to predict it
    """
    return model.predict(X_test)
