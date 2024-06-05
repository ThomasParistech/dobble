# /usr/bin/python3
"""Profiling decorator"""
import inspect
import json
import os
import threading
import time
from collections.abc import Callable
from functools import wraps
from queue import Queue
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

PROFILE = True
PROFILING_T0: float = time.perf_counter()
PROFILING_EVENTS_QUEUE: Queue = Queue()  # [Tuple[str, str, float, float]]


def get_function_name(func: Callable):
    """Get file and function name"""
    module_name = func.__module__.split('.')[-1]
    if module_name == "__main__":
        module_name = os.path.basename(inspect.getfile(func))
    return f"{module_name}::{func.__name__}"


def push_profiling_event(name: str,  start_time: float, end_time: float, thread_id: Optional[str] = None):
    """Push profiling event"""
    if thread_id is None:
        thread_id = threading.current_thread().name
    # Queue is thread-safe
    PROFILING_EVENTS_QUEUE.put(
        (name, thread_id, start_time-PROFILING_T0, end_time-PROFILING_T0))


def profile(func: Callable):
    """Profiling decorator"""
    # Warning: This decorator won't work if the function runs inside a multiprocessing Process
    # Processes are not like threads; they do not share memory, which means the global variables are copied and not
    # modified outside the scope of the process
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not PROFILE:
            return func(*args, **kwargs)

        start_time = time.perf_counter()
        retval = func(*args, **kwargs)
        end_time = time.perf_counter()

        push_profiling_event(get_function_name(func), start_time, end_time)
        return retval
    return wrapper


def export_profiling_events(output_path: str):
    """Dump profiling events into a JSON file that can be provided to the Chrome Tracing Viewer"""
    if not PROFILE:
        return

    events: List[Dict[str, Union[str, int, float]]] = []
    while not PROFILING_EVENTS_QUEUE.empty():
        name, tid, t_begin, t_end = PROFILING_EVENTS_QUEUE.get()
        events.append({"name": name, "ph": "B",
                      "ts": t_begin*1e6, "tid": tid, "pid": 0})
        events.append({"name": name, "ph": "E",
                      "ts": t_end*1e6, "tid": tid, "pid": 0})

    dir_name = os.path.dirname(output_path)
    if dir_name != "":
        os.makedirs(dir_name, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as _f:
        json.dump({"traceEvents": events}, _f)

    print(
        f"Open Chrome, type chrome://tracing/ and load the file located at {os.path.abspath(output_path)}")


class LogScopeTime:
    """Log the time spent inside a scope. Use as context `with LogScopeTime(name):`."""

    def __init__(self, name: str):
        self._name = name

    def __enter__(self):
        self._start_time = time.perf_counter()
        print(self._name)

    def __exit__(self, exception_type, exception_value, exception_traceback):
        end_time = time.perf_counter()
        print(f"--- {(end_time - self._start_time): .2f} s ({self._name}) ---")
