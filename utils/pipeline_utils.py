import logging
from datetime import timedelta
from functools import partial
from timeit import default_timer as timer

import ray
from tqdm import tqdm

logger = logging.getLogger("root")


@ray.remote
def run(obj, *args, **kwargs):
    return obj.run_map(*args, **kwargs)


def initialize(**kwargs):
    if not ray.is_initialized():
        ray.init(**kwargs)


class TaskLogger:

    def __init__(self, func_name):
        self.func_name = func_name
        self._start = None
        self._end = None

    def start(self):
        logger.info(f"Starting: {self.func_name}")
        self._start = timer()

    def stop(self, additional=""):
        self._end = timer()
        logger.info(f"Finished: {self.func_name}. "
                    f"Time elapsed: {str(timedelta(seconds=self._end - self._start)).split('.')[0]}. "
                    f"{additional}")


class Task:

    def __init__(self, func, func_name, args, kwargs):
        self.func = func
        self.func_name = func_name
        self.func_partial = partial(func, *args, **kwargs)

    def run_map(self, *args, **kwargs):
        result = self.func(*args, **kwargs)
        return result

    def run(self):
        task_logger = TaskLogger(self.func_name)
        task_logger.start()
        result = self.func_partial()
        task_logger.stop()
        return result


class Map:

    def __init__(self, partial_func, task):
        self.partial_func = partial_func
        self.task = task()

    def run(self):
        task_logger = TaskLogger(self.task.func_name)
        task_logger.start()
        result = self.partial_func()
        task_logger.stop()
        return result


def task(func):
    def wrapper(*args, **kwargs):
        return Task(func, func.__name__, args, kwargs)

    return wrapper


def seq_map_(func, iterable, **kwargs):
    """
        Args:
        func: Function to be executed
        iterable: Iterable to be looped over
        **kwargs: Arguments of the function
    Returns:
        Result of the function
    """
    return [func().run_map(x, **kwargs) for x in tqdm(iterable)]


def seq_map(func, iterable, **kwargs):
    return Map(partial(seq_map_, func, iterable, **kwargs), func)


def seq_map_tuple_return(func, iterable, **kwargs):
    """
    Used for funcs that return several values (or a tuple of values). Note that any return
    value is a list. There is no need to call run().

    Args:
        func:
        iterable:
        **kwargs:

    Returns:

    """
    return map(list, zip(*seq_map(func, iterable, **kwargs).run()))


def par_map_(func, iterable, **kwargs):
    """
    Lot of overhead because of multiprocessing. Should only be used for processing intensive tasks.
    Args:
        func: Function to be executed
        iterable: Iterable to be looped over
        **kwargs: Arguments of the function
    Returns:
        Result of the function
    """
    result = [run.remote(func(), i, **kwargs) for i in iterable]
    return ray.get(result)


def par_map(func, iterable, **kwargs):
    return Map(partial(par_map_, func, iterable, **kwargs), func)


class FilterTask(Task):

    def __init__(self, func, func_name, args, kwargs):
        super().__init__(func, func_name, args, kwargs)
        self.in_len = len(args[0])

    def run(self):
        task_logger = TaskLogger(self.func_name)
        task_logger.start()
        result = self.func_partial()

        if isinstance(result, tuple):
            out_len = len(result[0])
        else:
            out_len = len(result)

        task_logger.stop(f"Dropped {self.in_len - out_len} items.")
        return result


def filter_task(func):
    def wrapper(*args, **kwargs):
        return FilterTask(func, func.__name__, args, kwargs)

    return wrapper
