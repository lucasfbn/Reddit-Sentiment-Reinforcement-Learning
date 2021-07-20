import logging
from functools import partial

import ray
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s %(asctime)s - %(message)s")
log = logging.getLogger()


@ray.remote
def run(obj, *args, **kwargs):
    return obj.run_map(*args, **kwargs)


def initialize():
    ray.init()


class Task:

    def __init__(self, func, func_name, args, kwargs):
        self.func = func
        self.func_name = func_name
        self.func_partial = partial(func, *args, **kwargs)

    def run_map(self, *args, **kwargs):
        result = self.func(*args, **kwargs)
        return result

    def run(self):
        logging.info(f"Starting: {self.func_name}")
        result = self.func_partial()
        logging.info(f"Finished: {self.func_name}")
        return result


class Map:

    def __init__(self, partial_func, task):
        self.partial_func = partial_func
        self.task = task()

    def run(self):
        logging.info(f"Starting: {self.task.func_name}")
        result = self.partial_func()
        logging.info(f"Finished: {self.task.func_name}")
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
