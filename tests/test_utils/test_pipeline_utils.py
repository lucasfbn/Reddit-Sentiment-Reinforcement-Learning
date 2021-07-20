import time

from utils.pipeline_utils import *

initialize()

input = list(range(1, 5))


@task
def loops(x, constant):
    time.sleep(0.1)
    return x + constant


@task
def loops_several_arguments(x, constant):
    time.sleep(0.1)
    return x + constant, x * constant


def test_task_exec():
    assert loops(1, 2).run() == 3


def test_task_sequential_exec():
    expected = [3, 4, 5, 6]

    assert seq_map(loops, input, constant=2).run() == expected


def test_task_parallel_exec():
    expected = [3, 4, 5, 6]

    assert par_map(loops, input, constant=2).run() == expected


def test_multiple_return_values_seq():
    x, y = map(list, zip(*seq_map(loops_several_arguments, input, constant=2).run()))

    assert x == [3, 4, 5, 6] and y == [2, 4, 6, 8]


def test_multiple_return_values_par():
    x, y = map(list, zip(*seq_map(loops_several_arguments, input, constant=2).run()))

    assert x == [3, 4, 5, 6] and y == [2, 4, 6, 8]
