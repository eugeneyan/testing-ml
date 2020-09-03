from functools import wraps
from time import perf_counter
from typing import Callable


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
def predict_with_time(model, X_test):
    return model.predict(X_test)
