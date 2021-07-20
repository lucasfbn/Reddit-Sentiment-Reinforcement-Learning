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

    def __init__(self, func, args, kwargs):
        self.func = func
        self.func_partial = partial(func, *args, **kwargs)

    def run_map(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def run(self):
        return self.func_partial()


class Map:

    def __init__(self, func):
        self.func = func

    def run(self):
        return self.func()


def task(func):
    logging.info(f"Starting: {func.__name__}")

    def wrapper(*args, **kwargs):
        return Task(func, args, kwargs)

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
    return Map(partial(seq_map_, func, iterable, **kwargs))


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
    return Map(partial(par_map_, func, iterable, **kwargs))
