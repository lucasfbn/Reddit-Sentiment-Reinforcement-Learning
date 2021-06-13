from prefect import task
from utils.mlflow_api import log_file
from typing import Tuple


@task
def mlflow_log_file(obj, fn):
    log_file(obj, fn)


@task
def unpack_union_mapping(mapping_result) -> Tuple[list, list]:
    """
    Used when we map a function that returns two results. The result of the mapping will be a list of tuples which are
    unpacked in this function.

    Args:
        mapping_result: The result of a mapping call which returns 2 results
    """
    result_len = len(mapping_result[0])
    result_1 = []
    result_2 = []

    for mr in mapping_result:
        result_1.append(mr[0])
        result_2.append(mr[1])

    return result_1, result_2


@task
def reduce_list(lst: list):
    """
    Reduces a list to the first element of the list. This is useful when we returned something from a mapped function
    where each element in the list is equal and we just need one of the elements.
    """
    return lst[0]
