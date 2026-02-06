import psutil
import os
import time
from functools import wraps
from types import FunctionType
from typing import Callable, Any, Dict, List

# Global dictionary to store stats
function_stats: Dict[str, List[Dict[str, float]]] = {}


def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)  # in MB


def track_stats(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        start_memory = get_memory_usage()

        result = func(*args, **kwargs)

        end_time = time.time()
        end_memory = get_memory_usage()

        stats = {
            "execution_time": end_time - start_time,
            "memory_used": end_memory - start_memory,
        }

        # Initialize list for this function if it doesn't exist
        func_name = func.__name__ if isinstance(func, FunctionType) else str(func)
        if func_name not in function_stats:
            function_stats[func_name] = []

        # Store the stats
        function_stats[func_name].append(stats)

        return result

    return wrapper
