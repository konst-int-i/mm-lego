import time

def timed_function(unit="s"):
    """
    Decorator to measure the runtime of a function
    Args:
        func: function to be measured

    Returns:
        Decorator that returns the elapsed time and the result of the function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            elapsed_time = end - start
            if unit == "m":
                elapsed_time /= 60
            return elapsed_time, result
        return wrapper
    return decorator