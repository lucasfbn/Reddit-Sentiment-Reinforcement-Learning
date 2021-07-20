import time

from utils.pipeline_utils import *

initialize()


@task
def loops(x, constant):
    time.sleep(0.1)
    return x + constant


def test_task_exec():
    assert loops(1, 2).run() == 3


def test_task_sequential_exec():
    input = list(range(1, 5))
    expected = [3, 4, 5, 6]

    assert seq_map(loops, input, constant=2).run() == expected


def test_task_parallel_exec():
    input = list(range(1, 5))
    expected = [3, 4, 5, 6]

    assert par_map(loops, input, constant=2).run() == expected
